FROM python:3.12-slim

# tesseract for screenshot OCR, poppler for rasterising scanned PDFs.
# Add more tesseract-ocr-<lang> packages if your customers write in other
# languages - OCR accuracy drops sharply on the wrong language model.
RUN apt-get update && apt-get install -y --no-install-recommends \
  tesseract-ocr poppler-utils libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
