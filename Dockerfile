# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents and requirements file into the container at /usr/src/app
COPY . .
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir flask
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/mobiusml/hqq.git@37502bea31f2969c6680c0c4a88ca74b3bb234a5


# Install huggingface-cli
RUN pip install huggingface-cli

# Copy the entrypoint script
COPY entrypoint.sh .
RUN chmod +x ./entrypoint.sh

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]

# Run server.py when the container launches
CMD ["python", "./server.py"]