FROM nvidia/cuda:10.0-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    bzip2 \
    curl \
    ca-certificates \
    gcc \
    git \
    htop \
    libx11-6 \
    nfs-common \
    sudo \
    vim \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /workspace
WORKDIR /workspace

# Install conda
RUN wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ./miniconda.sh
RUN chmod 0755 ./miniconda.sh
RUN bash ./miniconda.sh -b -p ./conda
ENV PATH="./conda/bin:$PATH"
RUN rm ./miniconda.sh

# Install project packages
COPY environment.yml ./
RUN conda update conda
RUN conda env create -f environment.yml
