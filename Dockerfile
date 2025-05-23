### Build image ###
FROM nvidia/cuda:12.6.1-devel-ubuntu24.04 AS build

# install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    software-properties-common \
    git \
    wget \
    libgles2-mesa-dev \
    libegl1 \
    xvfb \
    xauth \
    libspdlog-dev \
    libglm-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libtiff-dev \
    libzstd-dev \
    nasm \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libxkbcommon-x11-0

RUN apt-get install -y apt-transport-https ca-certificates gnupg

# get a current cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ noble main'
RUN apt-get install kitware-archive-keyring
RUN rm /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-get update && apt-get install -y cmake

# get python
RUN apt-get install -y python3-pip python3-venv

# get Qt installed
ENV QT_VERSION=6.8.3
RUN pip install aqtinstall --break-system-packages
RUN aqt install-qt --outputdir /qt linux desktop ${QT_VERSION} -m qtwebsockets qtimageformats
# required for qt offscreen platform plugin
RUN apt-get install -y libfontconfig
ENV QTDIR=/qt/${QT_VERSION}/gcc_64
ENV Qt6_DIR=/qt/${QT_VERSION}/gcc_64

# copy agave project
RUN mkdir /agave
COPY . /agave
RUN rm -rf /agave/build
RUN mkdir /agave/build
WORKDIR /agave

# install submodules
RUN git submodule update --init --recursive

# build agave project
RUN cd ./build && \
    cmake .. && \
    cmake --build . --config Release -j 8

# leaving this here to show how to load example data into docker image
# RUN mkdir /agavedata
# RUN cp AICS-11_409.ome.tif /agavedata/
# RUN cp AICS-12_881.ome.tif /agavedata/
# RUN cp AICS-13_319.ome.tif /agavedata/

EXPOSE 1235

COPY docker-entrypoint.sh /usr/local/bin/
RUN ["chmod", "+x", "/usr/local/bin/docker-entrypoint.sh"]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

#  docker build -t agave_latest .
#  docker run --name docker-agave -p 1235:1235 -d agave_latest