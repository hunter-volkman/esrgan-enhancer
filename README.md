🐳 How to Open Docker + Run the Project on MacBook
Assumes Docker Desktop is installed and running
Get it from: https://www.docker.com/products/docker-desktop/

✅ Step-by-Step
1. Navigate to Project

2. Build the Docker Image
`docker build -t esrgan-enhancer .`
This creates an image with Real-ESRGAN + Python + your scripts

3. Run the Container (No GPU, CPU-only mode)
`docker run -it -v $(pwd):/workspace esrgan-enhancer`
You are now inside the container shell!

4. Run an Example Inference
```
cd Real-ESRGAN
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i /workspace/test_images/example.jpg \
  -o /workspace/results \
  --outscale 2 --tile 512
```

Output will go to results/ folder back on your Mac.

🌐 Bonus: Run Streamlit on Mac (locally)
If you want to try the Streamlit app without Docker:
```
pip install streamlit realesrgan pillow
streamlit run streamlit_app.py
```
It will launch in your browser at http://localhost:8501.

🧪 Testing Suggestion
After everything is working on Mac:

Try a couple sample inferences

Test the run_inference.sh script:
`cd /workspace/scripts`
`./run_inference.sh /workspace/test_images /workspace/results`