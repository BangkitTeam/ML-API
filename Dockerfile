# Gunakan image Python resmi versi 3.11-slim 
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Install system dependencies untuk image processing dan membersihkan apt cache untuk mengurangi ukuran image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt terlebih dahulu agar Docker bisa memanfaatkan cache untuk dependencies
COPY requirements.txt .

# Install dependencies Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Expose port tempat Flask app akan dijalankan
EXPOSE 8080

# Command default untuk menjalankan Flask app dengan gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
