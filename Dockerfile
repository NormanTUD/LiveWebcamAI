FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl libgl1 libglib2.0-0 && \
    ln -sf python3 /usr/bin/python

# Copy app
WORKDIR /app
COPY . .

# Install Python deps
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 8000

CMD ["bash", "startup.sh"]
