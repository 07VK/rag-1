# --- Stage 1: Build Stage ---
# Use a full Python image to build dependencies, which may have system requirements.
FROM python:3.10 as builder

# Set the working directory
WORKDIR /app

# Install build dependencies if any (e.g., for packages that compile from source)
# RUN apt-get update && apt-get install -y build-essential

# Copy only the requirements file first to leverage Docker's layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# --- Stage 2: Final Runtime Stage ---
# Use the slim base image for the final, smaller application image.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Create a non-root user to run the application for better security
RUN addgroup --system app && adduser --system --group app
USER app

# Copy the installed packages from the builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy the application code
COPY . .

# Ensure the app user can run the server
ENV PATH="/home/app/.local/bin:${PATH}"

# Expose the port the app runs on
EXPOSE 8000

# The command to run when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]