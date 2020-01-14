### Build image ###
FROM ubuntu:18.04 as build

# install dependencies
RUN mkdir /agave && \
    mkdir /agave/build && \
    apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libboost-all-dev \
    libtiff-dev \
    libglm-dev \
    python-dev \
    libgl1-mesa-dev

# get Qt installed
ENV QT_VERSION_A=5.12
ENV QT_VERSION_B=5.12.5
ENV QT_VERSION_SCRIPT=5125
RUN wget https://download.qt.io/archive/qt/${QT_VERSION_A}/${QT_VERSION_B}/qt-opensource-linux-x64-${QT_VERSION_B}.run
RUN chmod +x qt-opensource-linux-x64-${QT_VERSION_B}.run 
COPY ci/qt-noninteractive.qs /qt-noninteractive.qs
RUN ./qt-opensource-linux-x64-${QT_VERSION_B}.run --script qt-noninteractive.qs -platform minimal

# copy agave project
COPY . /agave
WORKDIR /agave

# install submodules
RUN git submodule update --init --recursive

# build agentsim project
RUN cd ./build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make
