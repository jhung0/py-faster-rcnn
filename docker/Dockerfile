FROM nvidia/cuda:7.0-cudnn4-devel-ubuntu14.04

MAINTAINER Jane Hung <jyhung@mit.edu>

RUN apt-get update                                && \
    apt-get install -y \
      build-essential                                \
      ca-certificates                                \
      cmake                                          \
      curl                                           \
      debconf-utils                                  \
      gcc                                            \
      g++                                            \
      gfortran                                       \
      gnuplot                                        \
      git-all                                        \
      hdf5-tools                                     \
      libatlas-base-dev                              \
      libboost-all-dev                               \
      libbz2-dev                                     \
      libfftw3-dev                                   \
      libfreetype6-dev                               \
      libgflags-dev                                  \
      libgoogle-glog-dev                             \
      libgraphicsmagick1-dev                         \
      libhdf5-serial-dev                             \
      libjpeg8-dev                                   \
      libleveldb-dev                                 \
      liblmdb-dev                                    \
      libmysqlclient-dev                             \
      libncurses5-dev                                \
      libncursesw5-dev                               \
      libopencv-dev                                  \
      libpng12-dev                                   \
      libprotobuf-dev                                \
      libreadline-dev                                \
      libsdl2-dev                                    \
      libsnappy-dev                                  \
      libsox-dev                                     \
      libsox-fmt-all                                 \
      libsqlite3-dev                                 \
      libssl-dev                                     \
      libsuitesparse-dev                             \
      libtiff5-dev                                   \
      libxml2-dev                                    \
      libxslt1-dev                                   \
      libzmq3-dev                                    \
      llvm                                           \
      make                                           \
      nodejs-legacy                                  \
      npm                                            \
      openssh-server                                 \
      pkg-config                                     \
      protobuf-compiler                              \
      python-dev                                     \
      python-opencv				     \
      python-pip                                     \
      python-software-properties                     \
      python-vigra                                   \
      software-properties-common                     \
      swig					     \
      tk-dev                                         \
      tmux                                           \
      unzip                                          \
      vim                                            \
      wget                                           \
      xz-utils                                       \
      zlib1g-dev

RUN pip install --upgrade pip

RUN pip install Cython easydict

RUN git clone --recursive https://github.com/jhung0/py-faster-rcnn.git 

# Allow it to find CUDA libs
#RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && \
#ldconfig
RUN ls

WORKDIR py-faster-rcnn
RUN cd lib && make

RUN cd caffe-fast-rcnn/python && \
	for req in $(cat requirements.txt); do pip install $req; done

RUN cd caffe-fast-rcnn &&\ 
	sed 's/# WITH_PYTHON_LAYER := 1/WITH_PYTHON_LAYER := 1/' Makefile.config.example > Makefile.config #&&\ 
#	sed -i 's/# USE_CUDNN/USE_CUDNN/' Makefile.config #&&\ 

RUN ls
RUN cd caffe-fast-rcnn && make all -j8 && make pycaffe 

RUN mkdir /home/ubuntu && mkdir /home/ubuntu/try1 && mkdir output && mkdir output/faster_rcnn_end2end && mkdir output/faster_rcnn_end2end/train && mkdir /home/svg

ENTRYPOINT ["python", "run2stage.py"]
