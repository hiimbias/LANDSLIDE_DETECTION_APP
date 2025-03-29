import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def normalize_channel(channel):
    return ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)

def convert_hdf5_to_image(hdf5_path, output_path, format="PNG"):
    """ Convert an HDF5 image to PNG or JPG format. """
    
    with h5py.File(hdf5_path, 'r') as hdf:
        print("Keys in HDF5 file:", list(hdf.keys()))
        data = np.array(hdf.get('img'))  # Đọc ảnh từ file HDF5
        print("Image shape:", data.shape)  # Kiểm tra số lượng kênh

        # Kiểm tra nếu số kênh không đủ
        if data.shape[-1] < 4:
            print("Not enough channels in the image. At least 4 are needed (Red, Green, Blue, NIR).")
            return
        
        # Xử lý giá trị NaN đưa về giá trị nhỏ nhất
        data[np.isnan(data)] = 0.000001
        
        # Lấy các kênh màu
        data_red = data[:, :, 3]
        data_green = data[:, :, 2]
        data_blue = data[:, :, 1]

        # Chuẩn hóa các kênh về [0, 255]
        red_norm = normalize_channel(data_red)
        green_norm = normalize_channel(data_green)
        blue_norm = normalize_channel(data_blue)

        # Gộp thành ảnh RGB
        rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)

        # Chuyển sang dạng ảnh PIL và lưu lại
        img = Image.fromarray(rgb_image, 'RGB')
        img.save(output_path, format=format)
        print(f"Image successfully converted and saved as: {output_path}")

# Example usage
hdf5_path = r"C:\Users\PC\Downloads\archive (4)\TrainData\mask\mask_2.h5"
output_path = r"E:/Workspace/AI Projects/Landslide Detection/output_mask.png"  # hoặc "output_image.jpg"
convert_hdf5_to_image(hdf5_path, output_path, format="PNG")

