# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /code

# Copy requirements from the local folder
COPY requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files from local folder into container
COPY . /code/

# Expose port
EXPOSE 7860

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
