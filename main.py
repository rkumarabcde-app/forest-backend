import ee
import json
import os
from fastapi import FastAPI, HTTPException
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
    allow_origins=["*"],   
    allow_methods=["POST"],   
    allow_headers=["*"]
)

# --------------------------------------------------
# Tile Visualization Parameters
# --------------------------------------------------

VIS_PARAMS = {
    "palette": ["FF0000"],   # Red fill
    "opacity": 1          
}

# --------------------------------------------------
# Sentinel-2 Collection
# --------------------------------------------------

def get_sentinel(start, end, aoi):

    def mask_s2_clouds(image):
        scl = image.select("SCL")
        mask = (
            scl.neq(3)
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return image.updateMask(mask)
    region = aoi.bounds().buffer(20000)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
    )

    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel images found for selected dates")

    return (
        collection
        .median()
        .select(["B2","B3","B4", "B8"])
    )

def get_sentinel_ndvi (image, aoi):
        collection = image.clip(aoi).normalizedDifference(["B8", "B4"])
        return (
        collection
    )

# --------------------------------------------------
# Landsat NDVI Collection
# --------------------------------------------------

def get_landsat(start, end, aoi):

    def mask_landsat(image):
        qa = image.select("QA_PIXEL")
        cloud = qa.bitwiseAnd(1 << 3).neq(0)
        shadow = qa.bitwiseAnd(1 << 4).neq(0)
        return image.updateMask(cloud.Or(shadow).Not())

    def scale(image):
        optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
        return image.addBands(optical, overwrite=True)
    region = aoi.bounds().buffer(20000)
    collection = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"))
        .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterDate(start, end)
        .filterBounds(region)
        .map(mask_landsat)
        .map(scale)
    )
    return collection.select(["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5"]).median()

def landsat_ndvi(image, aoi):
    image1 = image.clip(aoi)
    ndvi_oli = image1.normalizedDifference(["SR_B5", "SR_B4"])
    ndvi_tm  = image1.normalizedDifference(["SR_B4", "SR_B3"])
    return ee.Image(
        ee.Algorithms.If(
            image1.bandNames().contains("SR_B5"),
            ndvi_oli,
            ndvi_tm
        )
    ).rename("NDVI")

# For imagery of base and comparison years
def get_rgb_image_url(year, start, end, aoi):
    if year >= 2017:
        img = get_sentinel(start, end, aoi)
        rgb = img.select(["B4", "B3", "B2"])
        vis = {"min": 0, "max": 3000}
    else:
        img = get_landsat(start, end, aoi)
        band_names = img.bandNames()
        rgb = ee.Image(
        ee.Algorithms.If(
            band_names.contains("SR_B4"),
            img.select(["SR_B4", "SR_B3", "SR_B2"]),  # L8/9
            img.select(["SR_B3", "SR_B2", "SR_B1"])   # L5/7
        )
        )
        vis = {"min": 0, "max": 0.3}

    return rgb.visualize(**vis).getMapId()["tile_fetcher"].url_format

# --------------------------------------------------
# Helper: Build Styled Tile Layer
# --------------------------------------------------

def get_tile_url(image):
    map_id = image.getMapId()
    return map_id["tile_fetcher"].url_format

# --------------------------------------------------
# API Endpoint
# --------------------------------------------------

from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    year1: int
    year2: int
    threshold: float
    geometry: dict
@app.post("/forest-loss")
async def forest_loss(data: AnalysisRequest):

    try:

        year1 = data.year1
        year2 = data.year2
        threshold = data.threshold
        geometry = data.geometry

        aoi = ee.Geometry(geometry)

        # ---------------------------
        # Date ranges (Oct–Nov window)
        # ---------------------------

        start1 = ee.Date.fromYMD(year1, 10, 1)
        end1   = ee.Date.fromYMD(year1, 11, 30)

        start2 = ee.Date.fromYMD(year2, 10, 1)
        end2   = ee.Date.fromYMD(year2, 11, 30)

        # ---------------------------
        # NDVI per year
        # ---------------------------

        def get_ndvi(year, start, end, aoi):
            if year >= 2017:
                img = get_sentinel(start, end, aoi)
                ndvi = get_sentinel_ndvi(img, aoi)
            else:
                img = get_landsat(start, end, aoi)
                ndvi = landsat_ndvi(img, aoi)
            return ndvi.rename("NDVI")

        ndvi1 = get_ndvi(year1, start1, end1, aoi)
        ndvi2 = get_ndvi(year2, start2, end2, aoi)

        ndvi_change = ndvi2.subtract(ndvi1)

        # ---------------------------
        # Forest loss mask
        # ---------------------------

        forest_loss_mask = ndvi_change.lt(-0.2).selfMask()

        # Connected components to isolate patches
        clusters = forest_loss_mask.connectedComponents(
            ee.Kernel.square(1), 512
        )
        labels = clusters.select("labels")

        # Mask patches smaller than threshold using pixelArea
        patch_area = ee.Image.pixelArea().updateMask(forest_loss_mask)

        patch_sum = patch_area.addBands(labels).reduceConnectedComponents(
            reducer=ee.Reducer.sum(),
            labelBand="labels"
        )
        large_patch_mask = patch_sum.gte(1000)
        forest_loss_mask = forest_loss_mask.updateMask(large_patch_mask)

        # Centroid generation
        # Add lon/lat bands to the masked patch image
        lonlat = ee.Image.pixelLonLat()

        # Compute the mean lon/lat within each connected patch
        centroids_image = lonlat.addBands(labels).reduceConnectedComponents(
            reducer=ee.Reducer.mean(),
            labelBand="labels"
        )

        # Apply the large-patch threshold mask
        centroids_image = centroids_image.updateMask(patch_sum.gte(threshold))

        # Sample the result to get a list of coordinates
        points = centroids_image.sample(
            region=aoi,
            scale = 10 if year2 >= 2017 else 30,
            geometries=True
        ).distinct(["longitude", "latitude"])

        points_geojson = points.getInfo()

        loss_filled = forest_loss_mask.gt(0).selfMask()

        #edge = loss_filled.focal_max(1).subtract(loss_filled.focal_min(1)).selfMask()

        styled_loss = loss_filled.visualize(palette=["FF0000"], opacity=1)
        #styled_outline = edge.visualize(palette=["FF0000"], opacity=1)

        #final_styled = ee.ImageCollection([styled_loss, styled_outline]).mosaic()

        # ---------------------------
        # Generate tile URLs — no getInfo() on vectors
        # ---------------------------

        tile_url = get_tile_url(styled_loss)
        base_rgb_tile = get_rgb_image_url(year1, start1, end1, aoi)
        comp_rgb_tile = get_rgb_image_url(year2, start2, end2, aoi)

        # ---------------------------
        # AOI bounds for map centering
        # (single small getInfo call — just 4 numbers)
        # ---------------------------

        bounds = aoi.bounds().getInfo()["coordinates"][0]
        lons   = [pt[0] for pt in bounds]
        lats   = [pt[1] for pt in bounds]

        return {
            "tile_url": tile_url,               # Forest loss
            "centroids": points_geojson,
            "base_rgb_tile": base_rgb_tile,
            "comp_rgb_tile": comp_rgb_tile,
            "year1": year1,
            "year2": year2,
            "bounds": {
                "min_lon": min(lons),
                "max_lon": max(lons),
                "min_lat": min(lats),
                "max_lat": max(lats),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
