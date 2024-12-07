# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir pymongo langchain openai numpy matplotlib pandas requests

# Expose a port (if your app runs a server, e.g., Flask)
EXPOSE 5000

# Command to run the app
CMD ["python", "StockSummaryBot.py"]