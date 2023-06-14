from fastapi import APIRouter, HTTPException, UploadFile, status
from extra_models import OCRModel, Base64PostModel, RestfulModel, resp_200, resp_400
from paddleocr import PaddleOCR
from utils.convert import base64_to_ndarray, bytes_to_ndarray

router = APIRouter(prefix="/ocr", tags=["OCR"])

ocr = PaddleOCR(use_angle_cls=True, lang="en")


@router.post('/predict-by-path', response_model=RestfulModel)
def predict_by_path(image_path: str):
    result = ocr.ocr(image_path, cls=True)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-base64', response_model=RestfulModel)
def predict_by_base64(base64model: Base64PostModel):
    img = base64_to_ndarray(base64model.base64_str)
    result = ocr.ocr(img, cls=True)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-file', response_model=RestfulModel)
async def predict_by_file(file: UploadFile):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):  # 只处理常见格式图片
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_data = file.file
        img = bytes_to_ndarray(file_data.read())
        result = ocr.ocr(img, cls=True)
        restfulModel.data = result
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image should be end with .jpg and .png"
        )
    return restfulModel