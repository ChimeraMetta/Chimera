# Stage 1: Base image with the specific PyPy version installed
FROM debian:bullseye-slim AS base

# Define PyPy version arguments for easy updates
ARG PYPY_VERSION_MAJOR_MINOR=3.8
ARG PYPY_VERSION=3.8-v7.3.7
ARG PYPY_TARBALL=pypy3.8-v7.3.7-linux64.tar.bz2
ARG PYPY_DOWNLOAD_URL=https://downloads.python.org/pypy/${PYPY_TARBALL}
ARG PYPY_INSTALL_PATH=/opt/pypy${PYPY_VERSION}

ENV PYTHONUNBUFFERED=1

# Install essential tools and runtime dependencies for PyPy
# ca-certificates: for HTTPS downloads
# wget/bzip2: for downloading/extracting PyPy
# libffi7, libssl1.1, zlib1g: Common runtime libraries needed by PyPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    bzip2 \
    libffi7 \
    libssl1.1 \
    zlib1g \
    # Clean up APT caches to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Download, extract, and install PyPy
RUN echo "Downloading PyPy from ${PYPY_DOWNLOAD_URL}" \
    && wget --progress=dot:giga ${PYPY_DOWNLOAD_URL} -O /tmp/${PYPY_TARBALL} \
    && mkdir -p ${PYPY_INSTALL_PATH} \
    && tar -xjvf /tmp/${PYPY_TARBALL} -C ${PYPY_INSTALL_PATH} --strip-components=1 \
    && rm /tmp/${PYPY_TARBALL}

# Ensure pip is installed and available for this PyPy version
RUN ${PYPY_INSTALL_PATH}/bin/pypy3 -m ensurepip --upgrade

# Create symbolic links to make this PyPy version the default python/pip
RUN ln -s ${PYPY_INSTALL_PATH}/bin/pypy3 /usr/local/bin/python \
    && ln -s ${PYPY_INSTALL_PATH}/bin/pypy3 /usr/local/bin/python3 \
    && ln -s ${PYPY_INSTALL_PATH}/bin/pip3 /usr/local/bin/pip \
    && ln -s ${PYPY_INSTALL_PATH}/bin/pip3 /usr/local/bin/pip3

# Verify installation
RUN python --version
RUN pip --version

# -----------------------------------------------------------
# Stage 2: Build the application using the PyPy base
FROM base AS builder

WORKDIR /app

# Copy requirements file first for Docker layer caching
# IMPORTANT: Make sure you have a 'requirements.txt' file in your project root!
COPY requirements.txt .

# Install project dependencies using PyPy's pip
# Add any build-time dependencies if needed by your packages (e.g., build-essential for C extensions)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the image
COPY . .

# Create a directory inside the container to mount the user's local code
# We set this as the working directory so relative paths work easily
RUN mkdir /workspace
WORKDIR /workspace

# Set the entrypoint to execute the CLI script using the installed PyPy
ENTRYPOINT ["python", "cli.py"]

# CMD can provide default arguments if needed, e.g., CMD ["--help"] 