from PIL import Image
import cv2
import numpy as np

from paddleocr import draw_ocr
from paddleocr import draw_structure_result, save_structure_res

font_path = './utils/fonts/simfang.ttf'

def draw_image(result, output_path, image):
    if len(result)==1:
        result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save('{}/result.jpg'.format(output_path))

def draw_pdf(result, output_path, images):
    for idx in range(len(result)):
        res = result[idx]
        image = imgs[idx]
        boxes = [line[0] for line in res]
        txts = [line[1][0] for line in res]
        scores = [line[1][1] for line in res]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        im_show = Image.fromarray(im_show)
        im_show.save('{}/result_{}.jpg'.format(output_path, idx))

def draw_structure(result, output_path, image):
    image = Image.open(img_path).convert('RGB')
    im_show = draw_structure_result(image, result,font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save('{}/result.jpg'.format(output_path))
