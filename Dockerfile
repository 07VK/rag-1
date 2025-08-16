# --- Stage 1: Build the React Frontend ---
FROM node:18-alpine as frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build


# --- Stage 2: Build the Python Backend ---
FROM python:3.10-slim
WORKDIR /app

# Install dependencies as root before switching user
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY app.py .

# Copy the built React app from the 'dist' folder to the 'static' folder
COPY --from=frontend-builder /app/frontend/dist ./static

# Create and switch to a non-root user
RUN addgroup --system app && adduser --system --group app
USER app

EXPOSE 8000
CMD ["uvicorn", "app.py:app", "--host", "0.0.0.0", "--port", "8000"]
