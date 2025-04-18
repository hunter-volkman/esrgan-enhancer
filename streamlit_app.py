# streamlit_app.py
import streamlit as st
from PIL import Image
import os
import subprocess
import tempfile

st.title("Real-ESRGAN Image Enhancer")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
scale = st.slider("Upscale Factor", 1, 4, 2)
tile = st.selectbox("Tile Size", [128, 256, 512], index=2)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.getbuffer())
        input_path = tmp.name
        output_dir = "streamlit_output"
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python", "inference_realesrgan.py",
            "-n", "RealESRGAN_x4plus",
            "-i", input_path,
            "-o", output_dir,
            "--tile", str(tile),
            "--outscale", str(scale)
        ]

        with st.spinner("Enhancing..."):
            subprocess.run(cmd)

        base = os.path.basename(input_path)
        out_path = os.path.join(output_dir, base.replace(".png", "_out.png"))

        if os.path.exists(out_path):
            st.image([input_path, out_path], caption=["Original", "Enhanced"], width=512)
        else:
            st.error("Failed to enhance image.")
