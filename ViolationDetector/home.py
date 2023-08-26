import streamlit as st
#from mmdet.apis import init_detector,inference_detector



st.title("Violation Detector! 📢")
st.sidebar.markdown("Home Page")
st.subheader("Choose any of the pages to navigate to from the sidebar! ")
st.subheader("You can choose any of the following violations to detect: ")
st.write("1. No walking zone violation")
st.write("2. No walking while texting/talking on phone")
st.write("3. Pets (cats/dogs) not allowed in the park")
st.write("4. Bicycles not allowed on the highway")

# def load_model():
#     config_file='/home/vishakha/Desktop/Project/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_fp16_1x_coco.py'
#     checkpoint_file='/home/vishakha/Desktop/Project/mmdetection/checkpoints/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth'
#     model=init_detector(config_file,checkpoint_file,device="cpu")
#     return model

