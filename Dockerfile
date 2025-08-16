# --- (Stage 1: frontend-builder remains the same) ---
FROM node:18-alpine as frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build


# --- Stage 2: Build the Python Backend ---
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Copy the built React app
COPY --from=frontend-builder /app/frontend/dist ./static

# --- FIX: ADD THIS LINE TO COPY YOUR IMAGES ---
COPY frontend/public/static ./static

# Create and switch to a non-root user
RUN addgroup --system app && adduser --system --group app
USER app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]