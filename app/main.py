from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from utils.metrics_initializer import recall_m, precision_m, f1_m
from local_logic import run_inference, save_h5_img
import h5py
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inference', methods=['POST'])
def inference():
    if 'image' not in request.files:
        return jsonify({"error": "anh chua duoc upload"}), 400

    # Lưu ảnh vào static
    image = request.files['image']
    
    if not os.path.exists("static"):
        os.makedirs("static")   
        
    image_path = os.path.join("static", image.filename)
    image.save(image_path)
    
    # Bắt đầu cho ảnh vào model
    input_path, output_path = run_inference(image_path)

    return render_template('display_images.html', input_image=input_path, output_image=output_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)