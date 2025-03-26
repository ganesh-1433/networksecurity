import sys
import os
import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI(
    title="Network Security Prediction API",
    description="API for training and predicting network security threats",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "", "last_column": ""})

@app.post("/train")
async def train_model(request: Request):
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        message = "✅ Model Training Completed Successfully!"
    except Exception as e:
        message = f"❌ Training Failed: {str(e)}"
    return templates.TemplateResponse("index.html", {"request": request, "message": message, "last_column": ""})

from fastapi.responses import RedirectResponse
from fastapi import Form

@app.post("/predict")
async def predict_model(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df['Predicted Output'] = y_pred
        df['Network Status'] = ["Safe" if pred == 0 else "Not Safe" for pred in y_pred]  

        last_column_df = df[['Predicted Output', 'Network Status']]

        print("\n📌 **Network Security Prediction Results**")
        print(last_column_df.to_string(index=False))

        request.app.state.last_result = last_column_df.to_html(classes='table table-bordered table-hover', index=False)

        return RedirectResponse(url="/result", status_code=303)

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "message": f"❌ Prediction Failed: {str(e)}"})

@app.get("/result")
async def show_results(request: Request):
    last_result = request.app.state.last_result if hasattr(request.app.state, 'last_result') else "<p>No results available.</p>"
    return templates.TemplateResponse("result.html", {"request": request, "last_column": last_result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
