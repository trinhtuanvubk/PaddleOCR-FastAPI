from PIL import Image
import cv2
import numpy as np
import fitz
import os

from paddleocr import draw_ocr
from paddleocr import draw_structure_result, save_structure_res

font_path = './utils/fonts/simfang.ttf'

def save_img(img, output_path, filename):
    # image(Image|array): RGB image
    im_show = Image.fromarray(img)
    save_path = '{}/{}'.format(output_path, filename)
    im_show.save(save_path)
    return save_path

def save_pdf(file, output_path, filename):
    save_path = '{}/{}'.format(output_path, filename)
    with open(save_path, "wb") as f:
        f.write(file.read())
    
    return save_path

def draw_image(result, output_path, image, filename):
    if len(result)==1:
        result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    save_path = '{}/{}'.format(output_path, filename)
    im_show.save(save_path)

    return save_path

def draw_pdf(result, output_path, image_path, filename):
    imgs = []
    output_dir_path = os.path.join(output_path, filename.split(".")[0])
    os.makedirs(output_dir_path, exist_ok=True)
    with fitz.open(image_path) as pdf:
        for pg in range(0, pdf.pageCount):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.getPixmap(matrix=mat, alpha=False)
            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)
    for idx in range(len(result)):
        res = result[idx]
        image = imgs[idx]
        boxes = [line[0] for line in res]
        txts = [line[1][0] for line in res]
        scores = [line[1][1] for line in res]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        im_show = Image.fromarray(im_show)
        img_output_path = os.path.join(output_dir_path, "{}.jpg".format(idx))
        im_show.save(img_output_path)
    return output_dir_path

def draw_structure(result, output_path, image):
    image = Image.open(img_path).convert('RGB')
    im_show = draw_structure_result(image, result,font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save('{}/result.jpg'.format(output_path))
