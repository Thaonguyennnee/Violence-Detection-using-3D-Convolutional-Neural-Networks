FROM python:3.8

WORKDIR /C:/Users/LAPTOP/OneDrive/Tài liệu/2023/FALL/OJT/ThaoNguyen/video-recognition

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt .
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000


CMD ["python", "app.py"]
