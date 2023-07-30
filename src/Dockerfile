# Use the official Python image as the base image
FROM python:3.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create a directory inside the container and set it as the working directory
WORKDIR /app

# Copy the FastAPI application files, requirements.txt and the best model path to the container
COPY app.py run_server.py requirements.txt best_model_weights.pth ./

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI server when the container is run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]