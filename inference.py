#!/usr/bin/env python3

import cv2
from matplotlib.pyplot import fill
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw, Image, ImageFont
import time
import sys, os


import common
from downloader import getFilePath
from locate_and_correct import locate_and_correct
from data_processing import PreprocessYOLO

########################22
# Common
########################
cap = cv2.VideoCapture(0)
unet_engine_file_path = "unet_fp32_new.trt"
cnn_blue_engine_file_path = "cnn_blue_fp16.trt"
cnn_yellow_engine_file_path = "cnnyellow_fp32.trt"
cnn_green_engine_file_path = "cnn_green_fp32.trt"
font = ImageFont.truetype("SimHei.ttf", 20, encoding="utf-8")
characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
                "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

########################
# 获取TRT
########################
TRT_LOGGER = trt.Logger()


def judge_color(img):
    """
    :param: img:opencv/np.ndarray
    :returns: color: str
    """
    green = yello = blue = black = white = 0
    card_img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    row_num, col_num = card_img_hsv.shape[:2]
    card_img_count = row_num * col_num
    color = "no"

    for i in range(row_num):
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if 11 < H <= 34 and S > 34:
                yello += 1
            elif 35 < H <= 99 and S > 34:
                green += 1
            elif 99 < H <= 124 and S > 34:
                blue += 1
    if yello * 2 >= card_img_count:
        color = "yellow"
    elif green * 2 >= card_img_count:
        color = "green"
    elif blue * 2 >= card_img_count:
        color = "blue"

    return color

def image_resize(raw_image):
    """Load an image from the specified path and resize it to the input resolution.
    Return the input image before resizing as a PIL Image (required for visualization),
    and the resized image as a NumPy float array.

    Keyword arguments:
    input_image_path -- string path of the image to be loaded
    """

    image_raw = raw_image
    # Expecting yolo_input_resolution in (height, width) format, adjusting to PIL
    # convention (width, height) in PIL:
    new_resolution = (
        240, # w
        80) # h
    image_resized = image_raw.resize(
        new_resolution, resample=Image.BICUBIC)
    # image_resized.show()
    image_resized = np.array(image_resized, dtype=np.float32, order='C')
    return image_resized

def get_engine(engine_file_path=""):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else :
        print("ERROR! Can't find engine")

def main():
    COUNT = 0
    FPS = 30
    ##############################################
    # Init Engine And Context
    ##############################################
    unet_engine  = get_engine(unet_engine_file_path)
    unet_context = unet_engine.create_execution_context()
    unet_inputs, unet_outputs, unet_bindings, unet_stream = common.allocate_buffers(unet_engine)
    
    # 这里可以使用数组和循环进行优化
    cnn_blue_engine  = get_engine(cnn_blue_engine_file_path)
    cnn_blue_context = cnn_blue_engine.create_execution_context()
    cnn_blue_inputs, cnn_blue_outputs, cnn_blue_bindings, cnn_blue_stream = common.allocate_buffers(cnn_blue_engine)    

    cnn_yellow_engine  = get_engine(cnn_yellow_engine_file_path)
    cnn_yellow_context = cnn_yellow_engine.create_execution_context()
    cnn_yellow_inputs, cnn_yellow_outputs, cnn_yellow_bindings, cnn_yellow_stream = common.allocate_buffers(cnn_yellow_engine)  
    
    cnn_green_engine  = get_engine(cnn_green_engine_file_path)
    cnn_green_context = cnn_green_engine.create_execution_context()
    cnn_green_inputs, cnn_green_outputs, cnn_green_bindings, cnn_green_stream = common.allocate_buffers(cnn_green_engine)  
    
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    input_image_path = '/home/asklv/downloads/LPR/Final_copy/trt/1071.png'

    ##############################################
    # UNet Inference Start
    ##############################################
    input_size = (512, 512)
    preprocessor = PreprocessYOLO(input_size)

    while True:
        start_time = time.time()
        
        success, img = cap.read()

        image_raw, image = preprocessor.process(img)
        
        # Load UNet Engine
        trt_outputs = []
        # Do inference
        print('Running UNet inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        unet_inputs[0].host = image
        trt_outputs = common.do_inference_v2(unet_context, 
                                            bindings = unet_bindings,
                                            inputs   = unet_inputs,
                                            outputs  = unet_outputs,
                                            stream   = unet_stream)
        ##############################################
        # UNet Inference End
        ##############################################
        unet_time = time.time()
        
        # Generate mask
        img_mask = np.array(trt_outputs)
        img_mask = img_mask.reshape(512, 512, 3)
        img_mask = img_mask / np.max(img_mask) * 255
        img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]
        img_mask = img_mask.astype(np.uint8)

        # correct
        Lic_img, bbox = locate_and_correct(image_raw, img_mask)
        
        
        # unet_time = time.time()
        print("bbox: {}, unet_time: {}, unet_fps: {}".format(bbox, unet_time - start_time, 1//(unet_time - start_time)))
        
        image_raw = image_raw.astype(np.uint8)
        image_raw = Image.fromarray(image_raw)
        draw = ImageDraw.Draw(image_raw)
        
        # 遍历Lic
        for lic in Lic_img:
            # print(type(lic))
            # color = judge_color(lic)
            # print(color)
            lic = lic.astype(np.uint8)
            # 选择颜色
            color = judge_color(lic)
            lic = Image.fromarray(lic.astype(np.uint8))     
            lic = image_resize(lic)
            lic = np.expand_dims(lic, axis=0) # 1, 80, 240, 3


            ##############################################
            # CNN Inference Start
            ##############################################
            # Load CNN Engine
            trt_outputs = []
            # Load CNN Engine
            # Do inference
            print('Running CNN inference on image {} color: --- {}...'.format(input_image_path, color))
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            if(color == "blue"):
                cnn_blue_inputs[0].host = lic
                trt_outputs = common.do_inference_v2(cnn_blue_context,
                                                    bindings = cnn_blue_bindings,
                                                    inputs   = cnn_blue_inputs,
                                                    outputs  = cnn_blue_outputs,
                                                    stream   = cnn_blue_stream)
            elif(color == "yellow"):
                # !!!注意，这里的参数写错了！！！
                cnn_yellow_inputs[0].host = lic
                trt_outputs = common.do_inference_v2(cnn_yellow_context,
                                                    bindings = cnn_yellow_bindings,
                                                    inputs   = cnn_yellow_inputs,
                                                    outputs  = cnn_yellow_outputs,
                                                    stream   = cnn_yellow_stream)
            elif(color == "green"):
                cnn_green_inputs[0].host = lic
                trt_outputs = common.do_inference_v2(cnn_green_context,
                                                    bindings = cnn_green_bindings,
                                                    inputs   = cnn_green_inputs,
                                                    outputs  = cnn_green_outputs,
                                                    stream   = cnn_green_stream)

            # Decode
            license  = ''
            if (color == "blue" or color == "yellow"):
                lic_pred = np.array(trt_outputs).reshape(7, 65)  # 列表转为ndarray，形状为(7,65)
                for arg in np.argmax(lic_pred, axis=1): # 取每行中概率值最大的arg, 将其转为字符
                    license += characters[arg]
            elif (color == "green"):
                lic_pred = np.array(trt_outputs).reshape(8, 65)  # 列表转为ndarray，形状为(8,65)
                for arg in np.argmax(lic_pred, axis=1): # 取每行中概率值最大的arg, 将其转为字符
                    license += characters[arg]

            ##############################################
            # CNN Inference End
            ##############################################

            end_time = time.time()
            print("===============================================================")
            print("======================  Predict Result  =======================")
            print("===============================================================")
            print("Source license: [{}] --- Predicted license: [{}]".format(input_image_path[0:-4], license))
            print("cost : {} --- fps : {}".format(end_time - unet_time, 1 / (end_time - unet_time)))
            print("---------------------------------------------------------------")
            draw.text((bbox[0], bbox[1] - 20),
                      str(license),
                      (255, 0, 0),
                      font=font)
            draw.rectangle([(bbox[0], bbox[1]),
                          (bbox[2], bbox[3])],
                          outline="red",
                          width=2)
            
        end_time = time.time()
        
        
        ##############################################
        # CNN Inference End
        ##############################################
        draw.text((10, 10),
        str("FPS: " + str(FPS)),
        (255, 0, 0),
        font=font)
        
        COUNT += 1
        if COUNT >= 5:
            FPS = 1 // (end_time - start_time)
            COUNT = 0
        
        
        ##############################################
        # CNN Inference End
        ##############################################
        image_raw = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        cv2.imshow("res", image_raw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()

"""
3.5
CNN部分耗时：
cost : 0.0015919208526611328 --- fps : 628.1719335030702
可以达到600多帧，所以这连个模型的速率及其不匹配，主要是前面的unet处理速率过慢，
所以在这里可以引入并行的unet，然后各自进行数据处理，这样可以极大提高速率，把unet的结果保存起来

Running UNet inference on image /home/asklv/downloads/LPR/Final_copy/trt/image_resized.jpg...
bbox: [195 394 328 429]
Running CNN inference on image /home/asklv/downloads/LPR/Final_copy/trt/image_resized.jpg...
===============================================================
======================  Predict Result  =======================
===============================================================
Source license: [/home/asklv/downloads/LPR/Final_copy/trt/image_resized] --- Predicted license: [贵AA8888]
cost : 0.012251138687133789 --- fps : 81.6250656806461
---------------------------------------------------------------
"""