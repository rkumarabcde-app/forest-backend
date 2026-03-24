import ee
import json
import os
import uuid
import time
import xml.etree.ElementTree as ET

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build

# --------------------------------------------------
# Environment & Earth Engine Initialization
# --------------------------------------------------

credentials_json = os.getenv("GEE_CREDENTIALS_JSON")
if not credentials_json:
    raise RuntimeError("GEE_CREDENTIALS_JSON not set")

credentials_dict = json.loads(credentials_json)

# GEE credentials
ee_credentials = ee.ServiceAccountCredentials(
    credentials_dict["client_email"],
    key_data=credentials_json
)
ee.Initialize(ee_credentials)

# Google Drive credentials (same service account)
# Make sure the service account has access to the Drive folder below
DRIVE_FOLDER_NAME = os.getenv("DRIVE_FOLDER_NAME", "forest_loss_exports")

drive_credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=["https://www.googleapis.com/auth/drive"]
)
drive_service = build("drive", "v3", credentials=drive_credentials)

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

def kml_to_ee_geometry(kml_path):
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
        if points[0] != points[-1]:
            points.append(points[0])
        polygons.append(points)

    if not polygons:
        raise ValueError("No polygon found in KML")

    if len(polygons) == 1:
        return ee.Geometry.Polygon(polygons[0])

    return ee.Geometry.MultiPolygon([[p] for p in polygons])

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

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
    )

    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel images found for selected dates")

    return collection.median().select(["B4", "B8"]).clip(aoi)

# --------------------------------------------------
# Landsat NDVI Collection
# --------------------------------------------------

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
        .map(scale)
        .map(add_ndvi)
    )

    return collection.select("NDVI").median().clip(aoi)

# --------------------------------------------------
# NDVI per year
# --------------------------------------------------

def get_image_for_year(year, start, end, aoi):
    if year >= 2017:
        img = get_sentinel(start, end, aoi)
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    else:
        ndvi = get_landsat_ndvi(start, end, aoi)
    return ndvi.rename("NDVI")

# --------------------------------------------------
# Drive helpers
# --------------------------------------------------

def get_or_create_drive_folder(folder_name: str) -> str:
    """Return the Drive folder ID, creating it if it doesn't exist."""
    query = (
        f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = drive_service.files().create(
        body=folder_metadata, fields="id"
    ).execute()
    return folder["id"]


def get_drive_file_url(filename: str, folder_id: str) -> str | None:
    """Return a public download URL for a file in the given Drive folder."""
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(
        q=query, fields="files(id, name, webContentLink)"
    ).execute()
    files = results.get("files", [])

    if not files:
        return None

    file_id = files[0]["id"]

    # Make file publicly readable
    drive_service.permissions().create(
        fileId=file_id,
        body={"role": "reader", "type": "anyone"},
    ).execute()

    return f"https://drive.google.com/uc?export=download&id={file_id}"


def delete_drive_file(filename: str, folder_id: str):
    """Clean up exported file from Drive after serving."""
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    for f in results.get("files", []):
        drive_service.files().delete(fileId=f["id"]).execute()

# --------------------------------------------------
# Endpoint 1: Submit export task → returns task_id
# --------------------------------------------------

@app.post("/forest-loss/submit")
async def submit_forest_loss(
    year1: int      = Form(...),
    year2: int      = Form(...),
    threshold: float = Form(...),
    kml: UploadFile  = File(...)
):
    if not kml.filename.endswith(".kml"):
        raise HTTPException(status_code=400, detail="Only KML files are allowed")

    # Save uploaded KML
    unique_name = f"{uuid.uuid4()}.kml"
    kml_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(kml_path, "wb") as buffer:
        buffer.write(await kml.read())

    try:
        aoi = kml_to_ee_geometry(kml_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Dynamic min_pixels
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

    # Date ranges
    start1 = ee.Date.fromYMD(year1, 10, 1)
    end1   = ee.Date.fromYMD(year1, 11, 30)
    start2 = ee.Date.fromYMD(year2, 10, 1)
    end2   = ee.Date.fromYMD(year2, 11, 30)

    # NDVI & change
    ndvi1      = get_image_for_year(year1, start1, end1, aoi)
    ndvi2      = get_image_for_year(year2, start2, end2, aoi)
    ndvi_change = ndvi2.subtract(ndvi1)

    # Loss detection
    scale      = 10 if year2 >= 2017 else 30
    forest_loss = ndvi_change.lt(-0.2).selfMask()
    connected   = forest_loss.connectedPixelCount(500, True)
    forest_loss = forest_loss.updateMask(connected.gte(min_pixels))

    clusters = forest_loss.connectedComponents(ee.Kernel.square(1), 512)

    vectors = clusters.select("labels").reduceToVectors(
        geometry=aoi,
        scale=scale,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e13
    )

    vectors = vectors.map(
        lambda f: f.set("area_m2", f.geometry().area(maxError=1))
    )

    significant_loss = vectors.filter(ee.Filter.gte("area_m2", threshold))
    centroids        = significant_loss.map(lambda f: f.centroid(maxError=1))

    # --------------------------------------------------
    # Unique filenames for this job
    # --------------------------------------------------
    job_id           = str(uuid.uuid4())[:8]
    loss_filename    = f"forest_loss_{job_id}"
    centroid_filename = f"centroids_{job_id}"

    folder_id = get_or_create_drive_folder(DRIVE_FOLDER_NAME)

    # --------------------------------------------------
    # Fire async export tasks (no .getInfo() on geometry!)
    # --------------------------------------------------
    loss_task = ee.batch.Export.table.toDrive(
        collection=significant_loss,
        description=loss_filename,
        folder=DRIVE_FOLDER_NAME,
        fileNamePrefix=loss_filename,
        fileFormat="GeoJSON"
    )
    loss_task.start()

    centroid_task = ee.batch.Export.table.toDrive(
        collection=centroids,
        description=centroid_filename,
        folder=DRIVE_FOLDER_NAME,
        fileNamePrefix=centroid_filename,
        fileFormat="GeoJSON"
    )
    centroid_task.start()

    return {
        "status": "submitted",
        "job_id": job_id,
        "loss_task_id": loss_task.id,
        "centroid_task_id": centroid_task.id,
        "poll_url": f"/forest-loss/status/{loss_task.id}/{centroid_task.id}/{job_id}"
    }

# --------------------------------------------------
# Endpoint 2: Poll task status → returns download URLs
# --------------------------------------------------

@app.get("/forest-loss/status/{loss_task_id}/{centroid_task_id}/{job_id}")
async def check_forest_loss_status(
    loss_task_id: str,
    centroid_task_id: str,
    job_id: str
):
    # Find tasks by ID
    all_tasks = ee.batch.Task.list()
    task_map  = {t.id: t for t in all_tasks}

    loss_task     = task_map.get(loss_task_id)
    centroid_task = task_map.get(centroid_task_id)

    if not loss_task or not centroid_task:
        raise HTTPException(status_code=404, detail="Task not found")

    loss_status     = loss_task.status()
    centroid_status = centroid_task.status()

    loss_state     = loss_status["state"]
    centroid_state = centroid_status["state"]

    # Still running
    if loss_state in ("READY", "RUNNING") or centroid_state in ("READY", "RUNNING"):
        return {
            "status": "processing",
            "loss_state": loss_state,
            "centroid_state": centroid_state
        }

    # Either task failed
    if loss_state == "FAILED" or centroid_state == "FAILED":
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "loss_error":     loss_status.get("error_message"),
                "centroid_error": centroid_status.get("error_message")
            }
        )

    # Both completed — fetch Drive URLs
    folder_id         = get_or_create_drive_folder(DRIVE_FOLDER_NAME)
    loss_filename     = f"forest_loss_{job_id}.geojson"
    centroid_filename = f"centroids_{job_id}.geojson"

    loss_url     = get_drive_file_url(loss_filename, folder_id)
    centroid_url = get_drive_file_url(centroid_filename, folder_id)

    if not loss_url or not centroid_url:
        return {"status": "processing", "note": "Files not yet visible in Drive"}

    return {
        "status": "completed",
        "forest_loss_url": loss_url,
        "centroids_url":   centroid_url,
        "note": "Download the GeoJSON files from the URLs above"
    }
