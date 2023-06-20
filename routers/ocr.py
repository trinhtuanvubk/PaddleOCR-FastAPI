from fastapi import APIRouter, HTTPException, UploadFile, status
from extra_models import OCRModel, Base64PostModel, RestfulModel, resp_200, resp_400
from paddleocr import PaddleOCR
from utils.convert import base64_to_ndarray, bytes_to_ndarray, bytes_to_np
from utils.draw import draw_image, draw_pdf, draw_structure
from utils.draw import save_img, save_pdf

import os
import fitz
from PIL import Image

router = APIRouter(prefix="/ocr", tags=["OCR"])

ocr = PaddleOCR(use_angle_cls=True, lang="en")

api_input = "./api_input"
api_output = "./api_output"

# @router.post('/predict-by-path', response_model=RestfulModel)
# def predict_by_path(image_path: str):
#     result = ocr.ocr(image_path, cls=True)
#     restfulModel = RestfulModel(
#         resultcode=200, message="Success", data=result, cls=OCRModel)
#     return restfulModel


# @router.post('/predict-by-base64', response_model=RestfulModel)
# def predict_by_base64(base64model: Base64PostModel):
#     img = base64_to_ndarray(base64model.base64_str)
#     result = ocr.ocr(img, cls=True)
#     restfulModel = RestfulModel(
#         resultcode=200, message="Success", data=result, cls=OCRModel)
#     return restfulModel


@router.post('/predict-image-file', response_model=RestfulModel)
async def predict_image(file: UploadFile):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):
        file.file.seek(0)
        file_data = file.file
        img = bytes_to_ndarray(file_data.read())

        result = ocr.ocr(img, cls=True)
        # save input
        input_save_path = save_img(img, api_input, file.filename)
        # save output
        output_save_path = draw_image(result, api_output, img, file.filename)

        restfulModel.data = result
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image should be end with .jpg and .png"
        )
    return restfulModel

@router.post('/predict-pdf-file', response_model=RestfulModel)
async def predict_pdf(file: UploadFile):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith(".pdf"):
        file.file.seek(0)
        file_data = file.file
        input_save_path = save_pdf(file_data, api_input, file.filename)
        # img = bytes_to_np(file_data.read())
        result = ocr.ocr(input_save_path, cls=True)

        output_save_path = draw_pdf(result, api_output, input_save_path, file.filename)

        restfulModel.data = result
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image should be end with .pdf"
        )
    return restfulModel


# @router.post('/predict-image-file', response_model=RestfulModel)
# async def predict_image(file: UploadFile):
#     # restfulModel: RestfulModel = RestfulModel()
#     if file.filename.endswith((".jpg", ".png")):
#         file.file.seek(0)
#         file_data = file.file
#         img = bytes_to_ndarray(file_data.read())

#         result = ocr.ocr(img, cls=True)
#         # save input
#         input_save_path = save_img(img, api_input, file.filename)
#         # save output
#         output_save_path = draw_image(result, api_output, img, file.filename)

#         restfulModel.data = result
#         restfulModel.resultcode = 200
#         restfulModel.message = file.filename
#     else:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Image should be end with .jpg and .png"
#         )
#     return File

# @router.post('/predict-pdf-file', response_model=RestfulModel)
# async def predict_pdf(file: UploadFile):
#     restfulModel: RestfulModel = RestfulModel()
#     if file.filename.endswith(".pdf"):
#         file.file.seek(0)
#         file_data = file.file
#         input_save_path = save_pdf(file_data, api_input, file.filename)
#         # img = bytes_to_np(file_data.read())
#         result = ocr.ocr(input_save_path, cls=True)

#         output_save_path = draw_pdf(result, api_output, input_save_path, file.filename)

#         restfulModel.data = result
#         restfulModel.resultcode = 200
#         restfulModel.message = file.filename
#     else:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Image should be end with .pdf"
#         )
#     return restfulModel


'''
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

@app.get("/data")
async def get_data():
    # Generate your JSON data
    json_data = {"key": "value"}

    # Load your image
    image = Image.open("path/to/your/image.jpg")

    # Create an in-memory byte stream to store the image
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="JPEG")
    image_byte_array.seek(0)

    # Return the response with the image and JSON data
    return Response(content=image_byte_array, media_type="image/jpeg", headers={"Access-Control-Allow-Origin": "*"}, json=json_data)

'''