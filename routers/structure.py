from fastapi import APIRouter, HTTPException, UploadFile, status
from extra_models import OCRModel, Base64PostModel, RestfulModel, resp_200, resp_400
from paddleocr import PPStructure, save_structure_res
from utils.convert import base64_to_ndarray, bytes_to_ndarray, bytes_to_np
from utils.draw import draw_image, draw_pdf, draw_structure
from utils.draw import save_img, save_pdf

import os
import fitz
from PIL import Image

router = APIRouter(prefix="/structure", tags=["STRUCTURE"])

table_engine = PPStructure(show_log=True, lang='en')

api_input = "./api_input"
api_output = "./api_output"

@router.post('/predict-structure-image-file', response_model=RestfulModel)
async def predict_structure_image(file: UploadFile):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):
        file.file.seek(0)
        file_data = file.file
        img = bytes_to_ndarray(file_data.read())

        result = table_engine(img)
        # save structure res
        # save_folder = os.path.join(api_output, )
        save_structure_res(result, api_output, file.filename.split('.')[0])
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