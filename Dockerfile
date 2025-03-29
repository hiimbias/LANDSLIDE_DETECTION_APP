# Sử dụng image Python thuần thay vì uwsgi-nginx-flask
FROM python:3.11

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt thư viện cần thiết
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy project vào container
COPY ./requirements.txt /app/requirements.txt
COPY ./app /app/

# Cài đặt thư viện từ requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Mở cổng Flask
EXPOSE 5000

# Chạy Flask trực tiếp
CMD ["python", "main.py"]
