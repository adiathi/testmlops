FROM python:3.8-slim

WORKDIR /app

# Copy model and code
COPY catboost_model.cbm /app/
COPY requirements.txt /app/
COPY custom_model_server.py /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

CMD ["python", "custom_model_server.py"]
