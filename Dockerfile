### Build image ###
FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04 as build

# install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir /agave && \
    mkdir /agave/build && \
    apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    git \
    wget \
    libspdlog-dev \
    libtiff-dev \
    libglm-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1 \
    xvfb \
    xauth \
    libzstd-dev \
    nasm

# get a current cmake
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get install kitware-archive-keyring
RUN rm /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-get update && apt-get install -y cmake

# get python
RUN apt-get install -y python3.9-dev python3-pip
RUN pip3 install --upgrade pip

# get Qt installed
ENV QT_VERSION=6.5.0
RUN pip3 install aqtinstall
RUN aqt install-qt --outputdir /qt linux desktop ${QT_VERSION} -m qtwebsockets qtimageformats
# required for qt offscreen platform plugin
RUN apt-get install -y libfontconfig

# copy agave project
COPY . /agave
RUN rm -rf /agave/build/*
WORKDIR /agave

# install submodules
RUN git submodule update --init --recursive

# build agave project
ENV QTDIR=/qt/${QT_VERSION}/gcc_64
ENV Qt6_DIR=/qt/${QT_VERSION}/gcc_64
RUN cd ./build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make

# leaving this here to show how to load example data into docker image
# RUN mkdir /agavedata
# RUN cp AICS-11_409.ome.tif /agavedata/
# RUN cp AICS-12_881.ome.tif /agavedata/
# RUN cp AICS-13_319.ome.tif /agavedata/

EXPOSE 1235

COPY docker-entrypoint.sh /usr/local/bin/
RUN ["chmod", "+x", "/usr/local/bin/docker-entrypoint.sh"]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]