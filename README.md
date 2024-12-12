# ML-API
API to predict trash classification from machine learning model

## How to RUN ML API in Lokal with Docker?
1. Build Docker Container Images
   ```
   Docker build -t flask-api .
   ```
2. Run docker container
   ```
   docker run -p 8080:8080 flask-api
   ``` 

## How to Deploy to Cloud run?
1. Set up your in terminal
   ```
   gcloud init
   ```
2. Deploy to your cloud account
   ```
   gcloud builds submit --config cloudbuild.yaml .
   ```
