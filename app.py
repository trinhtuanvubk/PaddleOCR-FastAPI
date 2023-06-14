from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from extra_models import *
from routers import ocr
from utils.convert import *

app = FastAPI(title="PaddleOCR-API")

origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ocr.router)