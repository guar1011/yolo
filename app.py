import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import pandas as pd

st.title("🌊 Under the sea detection")

st.write("Upload your Image...")

model_path = "models/best.pt"  # เส้นทางไฟล์โมเดล best.pt
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

@st.cache(allow_output_mutation=True)
def load_model():
    return model

def count_objects_left_side(image, model):
    # แบ่งรูปภาพเป็นครึ่งหนึ่ง
    height, width = image.shape[:2]
    half_width = width // 2
    left_image = image[:, :half_width]

    # ทำการตรวจจับวัตถุหรือจุดสำคัญบนรูปภาพฝั่งซ้าย
    result = model(left_image, size=600)
    detect_class = result.pandas().xyxy[0]
    
    # นับจำนวนวัตถุทั้งหมด
    object_count = len(detect_class)

    # สร้าง DataFrame สำหรับเก็บข้อมูลวัตถุ
    objects_df = pd.DataFrame(detect_class, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"])
    objects_df = objects_df[["name", "xmin", "ymin", "xmax", "ymax"]]

    # แสดงผลลัพธ์
    st.write("Total Objects:", object_count)
    st.write("Objects on the Left Side:", object_count)
    st.dataframe(objects_df)

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()))
    image = cv2.imdecode(file_bytes, 1)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(imgRGB)

    st.write("")
    st.write("Detecting...")

    model = load_model()
    count_objects_left_side(imgRGB, model)
