import ee
import json
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Environment & Earth Engine Initialization
# --------------------------------------------------

credentials_json = os.getenv("GEE_CREDENTIALS_JSON")

if not credentials_json:
    raise RuntimeError("GEE_CREDENTIALS_JSON not set")

credentials_dict = json.loads(credentials_json)

credentials = ee.ServiceAccountCredentials(
    credentials_dict["client_email"],
    key_data=credentials_json
)

ee.Initialize(credentials)

# --------------------------------------------------
# FastAPI Setup
# --------------------------------------------------

app = FastAPI(title="Forest Loss API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------
# Helper: Convert KML to EE Geometry
# --------------------------------------------------

import xml.etree.ElementTree as ET

def kml_to_ee_geometry(kml_path):

    tree = ET.parse(kml_path)
    root = tree.getroot()

    # KML namespace handling
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    polygons = []

    # Find all Polygon coordinate tags
    for coords in root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns):

        coord_text = coords.text.strip()
        points = []

        for line in coord_text.split():
            lon, lat, *_ = line.split(",")
            points.append([float(lon), float(lat)])

        # Ensure polygon is closed
        if points[0] != points[-1]:
            points.append(points[0])

        polygons.append(points)

    if not polygons:
        raise ValueError("No polygon found in KML")

    # If single polygon
    if len(polygons) == 1:
        return ee.Geometry.Polygon(polygons[0])

    # If multiple polygons → MultiPolygon
    return ee.Geometry.MultiPolygon([ [p] for p in polygons ])

# --------------------------------------------------
# Sentinel-2 Collection
# --------------------------------------------------

def get_sentinel(start, end, aoi):

    def mask_s2_clouds(image):
        scl = image.select("SCL")
        mask = (
            scl.neq(3)   # cloud shadow
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))  # clouds
        )
        return image.updateMask(mask)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
    )

    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel images found for selected dates")

    return (
        collection
        .median()
        .select(["B4", "B8"])
        .clip(aoi)
    )

def get_landsat_ndvi(start, end, aoi):

    def mask_landsat(image):
        qa = image.select("QA_PIXEL")
        cloud = qa.bitwiseAnd(1 << 3).neq(0)
        shadow = qa.bitwiseAnd(1 << 4).neq(0)
        return image.updateMask(cloud.Or(shadow).Not())

    def scale(image):
        optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
        return image.addBands(optical, overwrite=True)

    def add_ndvi(image):

        # Landsat 8/9
        ndvi_oli = image.normalizedDifference(["SR_B5", "SR_B4"])

        # Landsat 5/7
        ndvi_tm = image.normalizedDifference(["SR_B4", "SR_B3"])

        return ee.Image(
            ee.Algorithms.If(
                image.bandNames().contains("SR_B5"),
                ndvi_oli,
                ndvi_tm
            )
        ).rename("NDVI")

    collection = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"))
        .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterDate(start, end)
        .filterBounds(aoi)
        .map(mask_landsat)
        .map(scale)
        .map(add_ndvi)
    )

    return collection.select("NDVI").median().clip(aoi)

# --------------------------------------------------
# API Endpoint
# --------------------------------------------------

@app.post("/forest-loss")
async def forest_loss(
    year1: int = Form(...),
    year2: int = Form(...),
    threshold: float = Form(...),
    kml: UploadFile = File(...)
):

    try:

        if not kml.filename.endswith(".kml"):
            raise HTTPException(
                status_code=400,
                detail="Only KML files are allowed"
            )

        # ---------------------------
        # Save file securely
        # ---------------------------

        unique_name = f"{uuid.uuid4()}.kml"
        kml_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(kml_path, "wb") as buffer:
            buffer.write(await kml.read())

        aoi = kml_to_ee_geometry(kml_path)
        aoi_area_m2 = aoi.area(maxError=1)
        aoi_area_ha = aoi_area_m2.divide(10000)
        area_ha = aoi_area_ha.getInfo()

        if area_ha < 2000:
            min_pixels = 10      # small block / village
        elif area_ha < 10000:
            min_pixels = 30      # sub-district
        elif area_ha < 25000:
            min_pixels = 50
        elif area_ha < 50000:
            min_pixels = 100     # district
        else:
            min_pixels = 1000     # very large district/state

        # ---------------------------
        # Date ranges (Oct–Nov window)
        # ---------------------------

        start1 = ee.Date.fromYMD(year1, 10, 1)
        end1 = ee.Date.fromYMD(year1, 11, 30)

        start2 = ee.Date.fromYMD(year2, 10, 1)
        end2 = ee.Date.fromYMD(year2, 11, 30)

        # ---------------------------
        # NDVI
        # ---------------------------


        def get_image_for_year(year, start, end, aoi):

            if year >= 2017:
                img = get_sentinel(start, end, aoi)
                ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            else:
                ndvi = get_landsat_ndvi(start, end, aoi)

            return ndvi.rename("NDVI")

        ndvi1 = get_image_for_year(year1, start1, end1, aoi)
        ndvi2 = get_image_for_year(year2, start2, end2, aoi)

        ndvi_change = ndvi2.subtract(ndvi1)

        # ---------------------------
        # Loss detection
        # ---------------------------

        forest_loss = ndvi_change.lt(-0.2).selfMask()
        connected = forest_loss.connectedPixelCount(500, True)
        forest_loss = forest_loss.updateMask(connected.gte(min_pixels))

        # Connected components
        clusters = forest_loss.connectedComponents(
            ee.Kernel.square(1),
            512
        )

        vectors = clusters.select("labels").reduceToVectors(
            geometry=aoi,
            scale = 10 if year2 >= 2017 else 30,
            geometryType="polygon",
            eightConnected=True,
            maxPixels=1e13
        )

        # ---------------------------
        # Area filter (hectares)
        # ---------------------------

        vectors = vectors.map(
            lambda f: f.set(
                "area_m2",
                f.geometry().area(maxError=1)
            )
        )

        significant_loss = vectors.filter(
            ee.Filter.gte("area_m2", threshold)
        )

        centroids = significant_loss.map(
            lambda f: f.centroid(maxError=1)
        )

        # ⚠ Blocking but unavoidable if returning GeoJSON
        loss_geojson = vectors.getInfo()
        centroid_geojson = centroids.getInfo()

        return {
            "forest_loss": loss_geojson,
            "centroids": centroid_geojson
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
