# 🛰️ Hệ thống Phát hiện Sạt lở đất
Hệ thống áp dụng cấu trúc mô hình Unet vốn hoạt động tốt với tác vụ Phân đoạn ảnh (Image Segmentation) để phát hiện khu vực sạt lở đất thông qua ảnh vệ tinh

## 🚀 Giới Thiệu  
Dự án sử dụng dataset: https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense

Dự án được triển khai trên Docker

## 📌 Tính Năng  
- 🛠️ Phát hiện các khu vực sạt lở đất trong ảnh đầu vào và trả về ảnh các khu vực đó đã được phân đoạn và mask.

## 🖥️ Cài Đặt  
Hướng dẫn cách cài đặt dự án.
```sh
git clone https://github.com/username/repository.git
# Mở ứng dụng Docker Desktop
docker build -t landslide_detection_app .
docker-compose up 
``` 
Sau đó chạy ứng dụng trên http://localhost:5000/