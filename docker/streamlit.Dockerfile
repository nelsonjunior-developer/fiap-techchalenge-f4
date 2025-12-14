# docker/streamlit.Dockerfile
FROM python:3.11-slim

WORKDIR /app

# (Optional) system deps for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only the Streamlit entrypoint
COPY app.py ./app.py

# Health & networking
EXPOSE 8501
ENV PYTHONUNBUFFERED=1

# Default base URL for the API; Compose will override it at runtime
ENV API_BASE_URL=http://api:8000

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
