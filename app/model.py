import os
import numpy as np
import cv2
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.metrics_initializer import recall_m, precision_m, f1_m
from tensorflow.keras.models import load_model  # type: ignore

def load_trained_model(model_path="model/final_model_landslide_3.keras"):
    """
    Load the trained model directly from the saved .keras file.
    
    Args:
        model_path (str): Relative path to the saved model file.

    Returns:
        tf.keras.Model: Loaded model with custom metrics.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_model_path = os.path.join(base_dir, model_path)

    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found: {full_model_path}")

    try:
        model = load_model(
            full_model_path,
            custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m},
            safe_mode=False
        )
        print(f"Successfully loaded model from: {full_model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    return model




def process_image(h5_path, target_size=(128, 128)):
    """Tiền xử lý ảnh từ .h5 để phù hợp với mô hình TensorFlow."""
    with h5py.File(h5_path, "r") as hdf:
        data = np.array(hdf.get("img"))

        # Xử lý giá trị NaN
        data[np.isnan(data)] = 0.000001

        # Chuẩn hóa các kênh RGB
        mid_rgb = data[:, :, 1:4].max() / 2.0
        mid_slope = data[:, :, 12].max() / 2.0
        mid_elevation = data[:, :, 13].max() / 2.0

        # Tính NDVI
        data_red = data[:, :, 3]
        data_nir = data[:, :, 7]
        data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red), where=(data_nir + data_red) != 0)

        # Chuẩn bị input cho model (6 channels)
        input_data = np.zeros((1, target_size[0], target_size[1], 6))
        input_data[0, :, :, 0] = 1 - data[:, :, 3] / mid_rgb  # RED
        input_data[0, :, :, 1] = 1 - data[:, :, 2] / mid_rgb  # GREEN
        input_data[0, :, :, 2] = 1 - data[:, :, 1] / mid_rgb  # BLUE
        input_data[0, :, :, 3] = data_ndvi  # NDVI
        input_data[0, :, :, 4] = 1 - data[:, :, 12] / mid_slope  # SLOPE
        input_data[0, :, :, 5] = 1 - data[:, :, 13] / mid_elevation  # ELEVATION

        # Xử lý NaN lần cuối
        input_data[np.isnan(input_data)] = 0.000001
        return input_data
    


def predict_mask(model, input_path, threshold=0.5):
    """Dự đoán mask từ dữ liệu đã xử lý."""
    
    processed_image = process_image(input_path) 
    prediction = model.predict(processed_image)
    predicted_mask = (prediction > threshold).astype(np.uint8)
    
    return predicted_mask[0, :, :, 0] * 255 


    
def save_mask(mask, save_path):
    """Lưu mask dưới dạng ảnh PNG bằng OpenCV."""
    # Đảm bảo save_path có đuôi .png
    save_path = os.path.splitext(save_path)[0] + ".png"
    
    # Chuyển mask về uint8 nếu cần
    if mask.max() <= 1:
        mask = (mask * 255).astype("uint8")
    
    # Lưu ảnh dưới dạng PNG
    cv2.imwrite(save_path, mask)
    print(f"Mask đã được lưu tại: {save_path}")
    

if __name__ == "__main__":
    # Đường dẫn tới file .h5
    h5_path = r"E:\Workspace\AI Projects\v_test\static\image_2.h5"  # Thay đổi đường dẫn nếu cần
    model_path = "model/final_model_landslide_3.keras"  # Đường dẫn tới mô hình đã lưu

    # Tải mô hình đã huấn luyện
    model = load_trained_model(model_path)

    # Dự đoán mask
    mask = predict_mask(model, h5_path)

    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()
    # Lưu mask
    save_path = r"E:\Workspace\AI Projects\v_test\static\mask.png"  # Đường dẫn lưu mask
    save_mask(mask, save_path)
    
    
