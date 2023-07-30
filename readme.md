# About

Contains EDA, Training and a fully dockerized FastAPI backend for a article classifier.

# Test the Endpoints

### Hosted at http://54.254.8.19:8000/ (AWS EC2 - 2xlarge instance)

```
  curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "title": "Sample title text",
  "paragraph": "Sample paragraph content",
  "timestamp_seconds": 1669788240.0
  }' 
``` 


# File structure

### data - contains all the data files
### models - contains trained models
### notebooks - contains all the related notebooks
### src - contains the dockerized backend API code

# Steps to run

## uvicorn

- `pip install -r requirements.txt`
- `uvicorn run_server:app --reload`

## Docker

- `docker build -t article_classifier .`
- `docker run -d --name article_classifier -p 8000:8000 article_classifier`