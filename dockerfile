# Use an appropriate base image
FROM ubuntu:latest

# Update package lists and install libc6
RUN apt-get update && \
    apt-get install -y libc6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
