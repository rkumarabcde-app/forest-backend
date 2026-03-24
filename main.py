import ee
import json
import os
import uuid
import xml.etree.ElementTree as ET

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Environment & Earth Engine Initialization
# --------------------------------------------------

credentials_json = os.getenv("GEE_CREDENTIALS_JSON")
if not credentials_json:
    raise RuntimeError("GEE_CREDENTIALS_JSON environment variable not set")

credentials_dict = json.loads(credentials_json)

ee_credentials = ee.ServiceAccountCredentials(
    credentials_dict["client_email"],
    key_data=credentials_json
)
ee.Initialize(ee_credentials)

# --------------------------------------------------
# FastAPI Setup
# --------------------------------------------------

app = FastAPI(title="Forest Loss API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------
# Helper: Convert KML to EE Geometry
# --------------------------------------------------

def kml_to_ee_geometry(kml_path: str):
    """Parse a KML file and return an ee.Geometry (Polygon or MultiPolygon)."""
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    polygons = []

    for coords in root.findall(
        ".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns
    ):
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
        raise ValueError("No polygon found in KML file")

    if len(polygons) == 1:
        return ee.Geometry.Polygon(polygons[0])

    return ee.Geometry.MultiPolygon([[p] for p in polygons])

# --------------------------------------------------
# Sentinel-2 NDVI  (year >= 2017)
# --------------------------------------------------

def get_sentinel_ndvi(start, end, aoi):
    """Return a cloud-masked Sentinel-2 NDVI image for the given date range."""

    def mask_s2_clouds(image):
        scl = image.select("SCL")
        mask = (
            scl.neq(3)    # cloud shadow
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))  # clouds / cirrus
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
        raise ValueError("No Sentinel-2 images found for the selected date range")

    return (
        collection.median()
        .select(["B4", "B8"])
        .normalizedDifference(["B8", "B4"])
        .rename("NDVI")
        .clip(aoi)
    )

# --------------------------------------------------
# Landsat NDVI  (year < 2017)
# --------------------------------------------------

def get_landsat_ndvi(start, end, aoi):
    """Return a cloud-masked Landsat NDVI image (TM/ETM+/OLI) for the given date range."""

    def mask_landsat(image):
        qa = image.select("QA_PIXEL")
        cloud  = qa.bitwiseAnd(1 << 3).neq(0)
        shadow = qa.bitwiseAnd(1 << 4).neq(0)
        return image.updateMask(cloud.Or(shadow).Not())

    def scale_optical(image):
        optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
        return image.addBands(optical, None, True)

    def add_ndvi(image):
        # Landsat 8/9 → SR_B5 (NIR), SR_B4 (Red)
        # Landsat 5/7 → SR_B4 (NIR), SR_B3 (Red)
        ndvi_oli = image.normalizedDifference(["SR_B5", "SR_B4"])
        ndvi_tm  = image.normalizedDifference(["SR_B4", "SR_B3"])
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
        .map(scale_optical)
        .map(add_ndvi)
    )

    return collection.select("NDVI").median().clip(aoi)

# --------------------------------------------------
# Pick correct sensor per year
# --------------------------------------------------

def get_ndvi_for_year(year: int, start, end, aoi):
    """Return NDVI image using Sentinel-2 (>=2017) or Landsat (<2017)."""
    if year >= 2017:
        return get_sentinel_ndvi(start, end, aoi)
    return get_landsat_ndvi(start, end, aoi)

# --------------------------------------------------
# API Endpoint: Detect forest loss and return URLs
# --------------------------------------------------

@app.post("/forest-loss")
async def forest_loss(
    year1: int       = Form(...),
    year2: int       = Form(...),
    threshold: float = Form(...),
    kml: UploadFile  = File(...)
):
    """
    Detect forest loss between two years within the uploaded KML boundary.

    - year1      : Baseline year
    - year2      : Comparison year
    - threshold  : Minimum patch area in square metres (e.g. 5000 = 0.5 ha)
    - kml        : KML file defining the area of interest

    Returns GeoJSON download URLs that expire after ~1 hour.
    """

    # --------------------------------------------------
    # Validate file type
    # --------------------------------------------------
    if not kml.filename.endswith(".kml"):
        raise HTTPException(status_code=400, detail="Only KML files are accepted")

    # --------------------------------------------------
    # Save uploaded KML
    # --------------------------------------------------
    unique_name = f"{uuid.uuid4()}.kml"
    kml_path    = os.path.join(UPLOAD_DIR, unique_name)

    with open(kml_path, "wb") as f:
        f.write(await kml.read())

    try:
        aoi = kml_to_ee_geometry(kml_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --------------------------------------------------
    # Determine minimum connected pixels based on AOI size
    # --------------------------------------------------
    area_ha = aoi.area(maxError=1).divide(10000).getInfo()

    if area_ha < 2000:
        min_pixels = 10
    elif area_ha < 10000:
        min_pixels = 30
    elif area_ha < 25000:
        min_pixels = 50
    elif area_ha < 50000:
        min_pixels = 100
    else:
        min_pixels = 1000

    # --------------------------------------------------
    # Date windows: October – November
    # --------------------------------------------------
    start1 = ee.Date.fromYMD(year1, 10, 1)
    end1   = ee.Date.fromYMD(year1, 11, 30)
    start2 = ee.Date.fromYMD(year2, 10, 1)
    end2   = ee.Date.fromYMD(year2, 11, 30)

    # --------------------------------------------------
    # Compute NDVI for each year and derive change
    # --------------------------------------------------
    try:
        ndvi1 = get_ndvi_for_year(year1, start1, end1, aoi)
        ndvi2 = get_ndvi_for_year(year2, start2, end2, aoi)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ndvi_change = ndvi2.subtract(ndvi1).rename("NDVI_change")

    # --------------------------------------------------
    # Forest loss detection
    # --------------------------------------------------
    scale = 10 if year2 >= 2017 else 30

    # Pixels with NDVI drop > 0.2 are flagged as potential loss
    raw_loss = ndvi_change.lt(-0.2).selfMask()

    # Remove isolated pixels — keep connected patches >= min_pixels
    connected   = raw_loss.connectedPixelCount(500, True)
    forest_loss_img = raw_loss.updateMask(connected.gte(min_pixels))

    # Vectorize into polygon patches
    clusters = forest_loss_img.connectedComponents(ee.Kernel.square(1), 512)

    vectors = clusters.select("labels").reduceToVectors(
        geometry=aoi,
        scale=scale,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e13,
        reducer=ee.Reducer.countEvery()
    )

    # --------------------------------------------------
    # Filter patches by minimum area threshold
    # --------------------------------------------------
    vectors = vectors.map(
        lambda f: f.set("area_m2", f.geometry().area(maxError=1))
    )

    significant_loss = vectors.filter(ee.Filter.gte("area_m2", threshold))

    # Centroid of each loss patch
    centroids = significant_loss.map(lambda f: f.centroid(maxError=1))

    # --------------------------------------------------
    # Generate direct GeoJSON download URLs via GEE
    #
    # - No Drive export required
    # - No polling required — URLs returned immediately
    # - URLs expire after approximately 1 hour
    # - No server memory used (data streams directly from GEE)
    # --------------------------------------------------
    try:
        loss_url = significant_loss.getDownloadURL(filetype="geojson")
        centroid_url = centroids.getDownloadURL(filetype="geojson")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate download URLs: {str(e)}"
        )

    return {
        "status": "completed",
        "forest_loss_url": loss_url,
        "centroids_url": centroid_url,
        "area_ha": round(area_ha, 2),
        "note": "Download URLs expire after approximately 1 hour"
    }
