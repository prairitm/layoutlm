# Start with a Python base image
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Copy the rest of your application's code
COPY . /app
WORKDIR /app

CMD ["python", "train.py"]
