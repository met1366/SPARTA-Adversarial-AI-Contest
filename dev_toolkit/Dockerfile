FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL TASK="Sparta Project DockerFile"

# Install some basic utilities
RUN apt update && apt install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3.6\
    python3-pip\
 && rm -rf /var/lib/apt/lists/*

# Make python3 the default python version
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Copy the source code.
ADD . /app/

# Install the packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create a non-root user and switch to it
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user &&\
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user &&\
    chown -R user:user /app

USER user


ENTRYPOINT ["python", "main.py"]
CMD ["-h"]
