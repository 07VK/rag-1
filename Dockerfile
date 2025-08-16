# --- Stage 1: Build the React Frontend ---
# Use a Node.js base image to build the static assets
FROM node:18-alpine as frontend-builder

# Set the working directory for the frontend
WORKDIR /app/frontend

# Copy package.json and package-lock.json to leverage Docker caching
COPY frontend/package*.json ./
RUN npm install

# Copy the rest of the frontend code and build the static files
COPY frontend/ .
RUN npm run build


# --- Stage 2: Build the Python Backend ---
# Use a Python base image for the final application
FROM python:3.10-slim

# Set the working directory for the backend
WORKDIR /app

# Create a non-root user for better security
RUN addgroup --system app && adduser --system --group app
USER app

# Copy Python dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code (app.py)
COPY app.py .

# --- IMPORTANT ---
# Copy the built static frontend files from the first stage
# The 'build' folder from React will become the 'static' folder for FastAPI
COPY --from=frontend-builder /app/frontend/build ./static

# Expose the port the app runs on
EXPOSE 8000

# The command to run the Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]