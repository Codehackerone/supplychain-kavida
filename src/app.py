import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import get_prediction

app = FastAPI()

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Exception handler to handle HTTPExceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP Exception - {exc.status_code}: {exc.detail}")
    return {"error": exc.detail}

# Pydantic model to represent the structure of a NewsItem
class NewsItem(BaseModel):
    title: str
    paragraph: str
    timestamp_seconds: float

# Home route returning a simple message
@app.get('/')
def home():
    return {
        "message": "OK",
        "status": 200,
    }

# Prediction route to process a NewsItem and get the prediction output
@app.post("/predict/")
async def predict(news_item: NewsItem):
    # Extract data from the incoming request's NewsItem
    title = news_item.title
    paragraph = news_item.paragraph
    timestamp = news_item.timestamp_seconds
    
    # Call the 'get_prediction' function from the 'inference' module
    output = get_prediction(title, paragraph, timestamp)
    
    return output
