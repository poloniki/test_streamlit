import streamlit as st
import cv2


import streamlit as st
from PIL import Image

import cv2
import io
import time
import numpy as np




def load_images(cv_image):
    robo_prediction = cv_image
    st.image(robo_prediction)


def main():
    st.header('Head Hunter')
    st.markdown('Counting crowds with confidence since 2023.')
    st.markdown("---")


    confidence_threshold = st.sidebar.slider('Confidence threshold:', 0.0, 1.0, 0.3, 0.01)
    overlap_threshold = st.sidebar.slider('Overlap threshold:', 0.0, 1.0, 0.5, 0.01)


    img_file_buffer = st.file_uploader('')


    if img_file_buffer is not None:
        img_bytes = img_file_buffer.getvalue()
        # Load uploaded image
        pil_image = Image.open(io.BytesIO(img_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        st.image(pil_image)
        if st.button("Predict Please...."):
            load_images(cv_image)

if __name__ == '__main__':
    main()
