# --- Stage 1: Builder ---
# In this stage, we install all dependencies and train the model.
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy only the necessary files for training
COPY requirements.txt .
COPY main.py .
COPY config.py .
COPY smoking.csv .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the training script to generate the full_pipeline.joblib artifact
RUN python main.py


# --- Stage 2: Final Application ---
# In this stage, we build the lean, final image for the application.
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code from the current directory
COPY app.py .
COPY api.py .  
COPY prediction_service.py .
COPY forms.py .
COPY config.py .
COPY templates/ templates/

# Copy the trained model artifact from the 'builder' stage
COPY --from=builder /app/full_pipeline.joblib .

# Expose the port and run the application using Gunicorn
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]