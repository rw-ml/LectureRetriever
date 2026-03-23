FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip3 install --upgrade pip && pip3 install -r requirements.txt


# Copy source code
COPY main.py .
COPY api ./api/
COPY chunking ./chunking/
COPY database ./database/
COPY general_tools ./general_tools/
COPY pdf_preprocessing ./pdf_preprocessing/
COPY response_generation ./response_generation/


#directory for database
RUN mkdir -p /app/data



# Expose FastAPI port
EXPOSE 8000
#expose vllm port
EXPOSE 30001

# Run the API
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]