
import streamlit as st
from mmdet.apis import init_detector,inference_detector
import mmcv
import matplotlib.pyplot as plt
import cv2
import os
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd 


def load_model():
    config_file='/home/vishakha/Desktop/Project/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_fp16_1x_coco.py'
    checkpoint_file='/home/vishakha/Desktop/Project/mmdetection/checkpoints/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth'
    model=init_detector(config_file,checkpoint_file,device="cpu")
    return model


def infer_model1(img,list_coor,model):

    result=inference_detector(model,img)
    detections=[]
    
    #getting model results
    for box in result[0]:
      x1,y1,x2,y2=[int(i) for i in box[:4]] #coordinates
      conf=float(box[-1]) #confidence score
      if conf>0.5: #setting confidence threshold
        midx=x1+(x2-x1)/2
        midy=y2
        detections.append([x1,y1,x2,y2,conf,midx,midy])

    #checking for violations
    pts = np.array(list_coor, np.int32)
    pts = pts.reshape((-1,1,2))
    inside_or_out=[]
    for box in detections:
        x=box[-2]
        y=box[-1]
        inside_or_out.append(cv2.pointPolygonTest(pts, (x,y), False))

    violations=0
    for inorout,box in zip(inside_or_out,detections):
        x1,y1,x2,y2=[int(i) for i in box[:4]] 
        if inorout==-1.0: #outside the zone==>violation (in red)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
            violations=violations+1
        else:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)

    for i in range(0,len(list_coor)-1):
        start=list_coor[i]
        end=list_coor[i+1]
        cv2.line(img,start,end,(0,0,0),2)
    cv2.line(img,list_coor[-1],list_coor[0],(0,0,0),2)
    

    st.image(img)
    txt1 = "No. of zone violations: {fname}".format(fname = violations)
    st.subheader(txt1)


def draw_bound(drawing_mode,bg_image):

    list_coor=[]

    if bg_image is not None:

        original_img=np.asarray(Image.open(io.BytesIO(bg_image.getvalue())))
        img=cv2.resize(original_img,(800,650),cv2.INTER_AREA)
        height=img.shape[0]
        width=img.shape[1]

        canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)",  # Fixed fill color with some opacity
        stroke_width=1,
        stroke_color='#000000',
        background_color='#EEEEEE',
        background_image=Image.open(bg_image)  if bg_image else None,
        update_streamlit=False,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        point_display_radius=3 if drawing_mode == 'point' else 0,
        key="canvas",)

        if canvas_result.json_data is not None:
            #st.write(canvas_result.json_data["objects"])
            objects = pd.json_normalize(canvas_result.json_data['objects']) # need to convert obj to str because PyArrow
            obj_list=objects.values.tolist()

            if len(obj_list)>0:
                coor=obj_list[0][-1]
                coor.pop()
                list_coor=[]
                for item in coor:
                    x=item[-2]
                    y=item[-1]
                    list_coor.append([x,y])
                print("points: ",list_coor)
            return img,list_coor

    else:
        return None,None


def main():

    model=load_model()
    st.sidebar.markdown("1. People")
    st.title("No Walk Zone Violation 🧍")
    st.write("📍 The user specifies a safe zone by drawing a polygon and the violations are detected accordingly.")
    st.write("📍 This can be used in factories to identify safe zones and make sure workers are walking within them.")
    st.write("📍 This can also be used in zebra crossings to ensure pedestrian safety.")

    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("polygon","line", "rect", "circle","point"))
    
    bg_image = st.file_uploader("Upload image:", type=["png", "jpg"]) 
    image,list_coor=draw_bound(drawing_mode,bg_image)
    st.subheader("Violations in Red")

    result=st.button('Run on image')

    if result:
        st.write("Calculating results...")
        infer_model1(image,list_coor,model)



if __name__=='__main__':
    main()


