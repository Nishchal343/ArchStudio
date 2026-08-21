FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

COPY requirements.txt ./
# Install the CPU wheel explicitly.  Installing torch from the default PyPI
# index can select a CUDA build and pull several gigabytes of NVIDIA runtime
# packages even though this service has no GPU runtime.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY migrations ./migrations

RUN mkdir -p /app/storage/projects
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
