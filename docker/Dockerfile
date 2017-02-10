FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

MAINTAINER Jane Hung <jyhung@mit.edu>

RUN apt-get update                                && \
    apt-get install --no-install-recommends --yes    \
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

RUN mkdir -p /srv/src/ && cd /srv/src/            && \
    git clone https://github.com/yyuu/pyenv.git   && \
    cd pyenv/plugins/python-build                 && \
    ./install.sh

RUN python-build 2.7.12 /usr/local/               && \
    pip install --upgrade pip                     && \
    pip install --upgrade wheel                   && \
    pip install --upgrade                            \
      alembic                                        \
      amqp                                           \
      blaze                                          \
      boto                                           \
      celery                                         \
      cffi                                           \
      coverage                                       \
      cython                                         \
      dask                                           \
      easydict 					     \
      flask                                          \
      h5py                                           \
      imageio                                        \
      ipykernel                                      \
      ipyparallel                                    \
      joblib                                         \
      jsonschema                                     \
      jupyter                                        \
      keras                                          \
      line_profiler                                  \
      lxml                                           \
      memory_profiler                                \
      mock                                           \
      networkx                                       \
      nose                                           \
      numpy                                          \
      odo                                            \
      pandas                                         \
      psutil                                         \
      pyamg                                          \
      pyflakes                                       \
      pytest                                         \
      python-dateutil                                \
      pytz                                           \
      pyzmq                                          \
      redis                                          \
      requests                                       \
      scikit-image                                   \
      scikit-learn                                   \
      scipy                                          \
      seaborn                                        \
      simpleitk                                      \
      simplejson                                     

RUN git clone --recursive https://github.com/jhung0/py-faster-rcnn.git 

# Allow it to find CUDA libs
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && \
ldconfig

RUN cd py-faster-rcnn/lib && make

RUN cd py-faster-rcnn/caffe-fast-rcnn/python && \
	for req in $(cat requirements.txt); do pip install $req; done

RUN cd py-faster-rcnn/caffe-fast-rcnn &&\ 
	sed 's/# WITH_PYTHON_LAYER := 1/WITH_PYTHON_LAYER := 1/' Makefile.config.example > Makefile.config &&\ 
	sed -i 's/# USE_CUDNN/USE_CUDNN/' Makefile.config #&&\ 
#	sed -i 's/INCLUDE_DIRS := \$(PYTHON_INCLUDE) \/usr\/local\/include/INCLUDE_DIRS := \$(PYTHON_INCLUDE) \/usr\/local\/include \/usr\/include\/hdf5\/serial\//' Makefile.config &&\ 
#	sed -i 's/LIBRARY_DIRS := \$(PYTHON_LIB) \/usr\/local\/lib \/usr\/lib/LIBRARY_DIRS := \$(PYTHON_LIB) \/usr\/local\/lib \/usr\/lib \/usr\/lib\/x86_64-linux-gnu \/usr\/lib\/x86_64-linux-gnu\/hdf5\/serial/' Makefile.config &&\ 
#	sed -i 's/hdf5/hdf5_serial/g' Makefile &&\ 
#	sed -i 's/NVCCFLAGS += -ccbin=\$(CXX) -Xcompiler -fPIC \$(COMMON_FLAGS)/NVCCFLAGS += -D_FORCE_INLINES -ccbin=\$(CXX) -Xcompiler -fPIC \$(COMMON_FLAGS)/' Makefile &&\ 
#	cd /usr/lib/x86_64-linux-gnu &&\ 
#	ln -s libhdf5_serial.so.10.0.2 libhdf5.so &&\ 
#	ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so

RUN cd py-faster-rcnn/caffe-fast-rcnn && make all -j8 && make pycaffe 

RUN cd py-faster-rcnn && ls #python run2stage.py \
	--images test.txt \
	--cfg experiments/cfgs/faster_rcnn_end2end.yml \
	--model1 output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_lr0.01_iter_100000.caffemodel \ 
	--prototxt1 models/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt \ 
	--model2 caffe-fast-rcnn/models/bvlc_reference_caffenet/caffenet_train_iter_40000.caffemodel \ 
	--prototxt2 caffe-fast-rcnn/models/bvlc_reference_caffenet/deploy.prototxt \
	--output /home/ubuntu/svg 
