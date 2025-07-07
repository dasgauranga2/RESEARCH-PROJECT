FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    dkms \
    libglib2.0-0 \
    git \
    ca-certificates \
    build-essential \
    wget \
    bzip2 \
    ninja-build \
    jq \
    jp \
    tree \
    tldr \
    nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget https://dot.net/v1/dotnet-install.sh && \
    chmod +x dotnet-install.sh && \
    ./dotnet-install.sh --channel 7.0 && \
    ./dotnet-install.sh --channel 3.1 && \
    rm ./dotnet-install.sh

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

RUN apt-get install -y sudo

ARG USER_ID=1012
ARG GROUP_ID=1012

RUN groupadd -g $GROUP_ID gauranga && \
    useradd -u $USER_ID -g gauranga -m gauranga

RUN usermod -aG sudo gauranga

RUN echo "gauranga:gauranga" | chpasswd
USER gauranga
WORKDIR /home/gauranga

RUN conda init bash

SHELL [ "/bin/bash", "-c" ]

CMD ["bash"]