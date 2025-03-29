from model import load_trained_model, predict_mask, save_mask
import os
from PIL import Image
import h5py
import numpy as np

model = load_trained_model()

def normalize_channel(channel):
    return ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)

def save_h5_img(hdf5_path, format="PNG"):
    """ Convert an HDF5 image to PNG or JPG format and save to the static directory. """
    
    with h5py.File(hdf5_path, 'r') as hdf:
        print("Keys in HDF5 file:", list(hdf.keys()))
        data = np.array(hdf.get('img'))  # Read image from HDF5 file
        print("Image shape:", data.shape)  # Check the number of channels

        # Check if the number of channels is insufficient
        if data.shape[-1] < 4:
            print("Not enough channels in the image. At least 4 are needed (Red, Green, Blue, NIR).")
            return
        
        # Handle NaN values by setting them to the minimum value
        data[np.isnan(data)] = 0.000001
        
        # Extract color channels
        data_red = data[:, :, 3]
        data_green = data[:, :, 2]
        data_blue = data[:, :, 1]

        # Normalize channels to [0, 255]
        red_norm = normalize_channel(data_red)
        green_norm = normalize_channel(data_green)
        blue_norm = normalize_channel(data_blue)

        # Combine into an RGB image
        rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)

        # Convert to PIL image and save
        img = Image.fromarray(rgb_image, 'RGB')
        
        # Generate output path in the static directory
        output_filename = os.path.splitext(os.path.basename(hdf5_path))[0] + f".{format.lower()}"
        output_path = os.path.join("static", output_filename)
        
        img.save(output_path, format=format)
        print(f"Image successfully converted and saved as: {output_path}")
        
        return output_path
        

def run_inference(image_path):
    print(f"Running inference on: {image_path}")
    try:
        mask = predict_mask(model, image_path)
    except Exception as e:
        print(f"Error in predict_mask: {e}")
        return None, None

    original_path = save_h5_img(image_path)
    if original_path is None:
        print("Error in save_h5_img")
        return None, None

    output_path = os.path.join("static", "output_" + os.path.splitext(os.path.basename(image_path))[0] + ".png")
    
    try:
        save_mask(mask, output_path)
    except Exception as e:
        print(f"Error in save_mask: {e}")
        return None, None

    return original_path, output_path
