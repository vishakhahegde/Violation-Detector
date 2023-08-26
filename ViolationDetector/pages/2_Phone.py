import streamlit as st
from mmdet.apis import init_detector,inference_detector
import mmcv
import matplotlib.pyplot as plt
import cv2
import os
import io
import numpy as np
from PIL import Image
import pandas as pd 


def load_image():

    #uploading image
    st.write('Upload your image!')
    uploaded_file=st.file_uploader(label='Pick an image to test')

    #displaying image
    if uploaded_file is not None:
        image_data=uploaded_file.getvalue()
        st.image(image_data)
        return np.asarray(Image.open(io.BytesIO(image_data)))
    else:
        return None


def load_model():
    config_file='/home/vishakha/Desktop/Project/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_fp16_1x_coco.py'
    checkpoint_file='/home/vishakha/Desktop/Project/mmdetection/checkpoints/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth'
    model=init_detector(config_file,checkpoint_file,device="cpu")
    return model



def infer_model(img,model):

	img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	result=inference_detector(model,img)
	detections=[]
	for box in result[67]:
		x1,y1,x2,y2=[int(i) for i in box[:4]]
		conf=float(box[-1])
		if conf>0.4:
			detections.append([x1,y1,x2,y2,conf])
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),4)

	st.image(img[:,:,::-1])
	txt1 = "No. of violations: {fname}".format(fname = len(detections))
	st.subheader(txt1)
	

def main():
	
	model=load_model()
	st.sidebar.markdown("2. Phone")
	st.title("Texting/Talking on Phone Violation 📵")
	st.write("📍 Detects violations when a person is texting or talking on their phone while walking.")
	st.write("📍 Ensures worker safety in factories. ")
	st.write("📍 Can also be used in classrooms to catch students using their phones. ")

	image=load_image()
	result=st.button('Run on image')
	if result:
		st.write("Calculating results...")
		infer_model(image,model)


if __name__=="__main__":
	main()


