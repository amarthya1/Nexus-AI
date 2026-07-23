# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy requirements first to leverage Docker layer caching
COPY backend/requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy backend code into container
COPY backend/ /code/

# Expose Hugging Face Space default port
EXPOSE 7860

# Run FastAPI app with Uvicorn on port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
