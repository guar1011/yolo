import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch, json , cv2 , detect


st.title("Wrong sport detection")

st.write("Upload your Image...")

#model = torch.hub.load('./yolov5', 'custom', path='./best.pt', source='local')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt')

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
if uploaded_file is not None:
  
  file_bytes = np.asarray(bytearray(uploaded_file.read()))
  image = cv2.imdecode(file_bytes, 1)

  imgRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
  #st.image(imgRGB)

  st.write("")
  st.write("Detecting...")

  imgW = imgRGB.shape[1]
  Cimg = imgW/2
  
  result = model(imgRGB, size=600)
  
  detect_class = result.pandas().xyxy[0] 
  
  #labels, cord_thres = detect_class[:, :].numpy(), detect_class[:, :].numpy()
  
  #     xmin       ymin    xmax        ymax          confidence  class    name
  #0  148.605362   0.0    1022.523743  818.618286    0.813045      2      turtle
  
  st.code(detect_class[['name', 'xmin','ymin', 'xmax', 'ymax']])
  
  detectW = detect_class[detect_class['xmin'] <= Cimg]

  textA = 'นักเรียนทั้งหมด : ' + str(detect_class.shape[0]) + ' คน'
  textR = 'นักเรียนที่อยู่ผิดสนาม : ' + str(detectW.shape[0]) + ' คน'
  st.success(textA)
  st.success(textR)
  outputpath = 'output.jpg'
  
  result.render()  # render bbox in image
  for im in result.ims:
      im_base64 = Image.fromarray(im)
      im_base64.save(outputpath)
      img_ = Image.open(outputpath)
      st.image(img_, caption='Model Prediction(s)')
