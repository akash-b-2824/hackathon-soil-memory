FROM python:3.9-slim

WORKDIR /app

# 1. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy your script AND your CSV data files into the container
COPY . .

# 3. Expose the port Hugging Face requires
ENV PORT=7860
EXPOSE 7860

# 4. Start the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]