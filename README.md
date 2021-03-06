### Disclaimer

The official Faster R-CNN code (written in MATLAB) is available [here](https://github.com/ShaoqingRen/faster_rcnn).
If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn).

This repository contains a Python *reimplementation* of the MATLAB code.
This Python implementation is built on a fork of [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn).
There are slight differences between the two implementations.
In particular, this Python port
 - is ~10% slower at test-time, because some operations execute on the CPU in Python layers (e.g., 220ms / image vs. 200ms / image for VGG16)
 - gives similar, but not exactly the same, mAP as the MATLAB version
 - is *not compatible* with models trained using the MATLAB code due to the minor implementation differences
 - **includes approximate joint training** that is 1.5x faster than alternating optimization (for VGG16) -- see these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more information

# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)

This Python implementation contains contributions from Sean Bell (Cornell) written during an MSR internship.

Please see the official [README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more details.

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497) and was subsequently published in NIPS 2015.

### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

  You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
```/opt/caffe/python$ pip install -r requirements.txt ```

2. Python packages you might not have: `cython` (pip), `python-opencv` (sudo apt-get install python-opencv), `easydict` (pip/pip2.7)
When you can't do sudo, follow http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html but with
```cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL
_PREFIX=/home/jhung0/opencv/ .. ``` so then sudo in sudo make install is not necessary. Then add to pythonpath /home/jhung0/opencv/lib/python2.7/site-packages

3. [optional] MATLAB (required for PASCAL VOC evaluation only)

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training with VGG16, you'll need a K40 (~11G of memory)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/jhung0/py-faster-rcnn.git
  ```

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone Faster R-CNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `faster-rcnn` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place (copy from Makefile.config.example and uncomment WITH_PYTHON_LAYER := 1), then simply do:
    make -j8 && make pycaffe
    
    #if it doesn't work, try
    export PYTHONPATH=/path/to/caffe/python
    #then maybe recompile
    cd ~/caffe
    make all
    make pycaffe
    make test
    make runtest
    ```

5. Download pre-computed Faster R-CNN detectors
    ```Shell
    cd $FRCN_ROOT
    ./data/scripts/fetch_faster_rcnn_models.sh
    ```

    This will populate the `$FRCN_ROOT/data` folder with `faster_rcnn_models`. See `data/README.md` for details.
    These models were trained on VOC 2007 trainval.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

**Python**

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

I got some errors, so in 
```
$FCN_ROOT/lib/fast_rcnn/config.py
```
set 
```
__C.USE_GPU_NMS = False
```
and/or change
```
sm_35 into sm_30 in lib/setup.py
```
It worked using a GPU on AWS (https://github.com/rbgirshick/py-faster-rcnn/issues/2).

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit
	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```
6. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). ZF was trained at MSRA.

*Note: this does not work when I try to use them in training, so I did
```
wget http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_M_1024.caffemodel
```
and moved the file to data/imagenet_models/

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

If you see this error
```
EnvironmentError: MATLAB command 'matlab' not found. Please add 'matlab' to your PATH.
```
then you need to make sure the matlab binary is in your $PATH. MATLAB is currently required for PASCAL VOC evaluation.


This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.


###Extra: Train with other data
####Format Your Dataset

At first, the dataset must be well organzied with the required format.
```
try1
|-- data
    |-- Annotations
         |-- *.txt (Annotation files)
    |-- Images
         |-- *.png (Image files)
    |-- ImageSets
         |-- train.txt
```
The `train.txt` contains all the names(without extensions) of images files that will be used for training. For example, there are a few lines in `train.txt` below.

```
crop_000011
crop_000603
crop_000606
crop_000607
```
### Construct IMDB

You need to add a new python file describing the dataset we will use to the directory `$FRCNN_ROOT/lib/datasets`. Then the following steps should be taken.
  - Modify `self._classes` in the constructor function to fit your dataset.
  - Be careful with the extensions of your image files. See `image_path_from_index`.
  - Write the function for parsing annotations. See `_load_try1_annotation`.
  - Do not forget to add `import` syntaxes in your own python file and other python files in the same directory.

Then you should modify the `factory.py` in the same directory.

### Modify Prototxt

For example, if you want to use the model **VGG_CNN_M_1024**, then you should modify `train.prototxt` or `stage1_fast_rcnn_train.pt`, `stage1_rpn_train.pt`, `stage2_fast_rcnn_train.pt`, `stage2_rpn_train.pt`, and `faster_rcnn_test.pt` in `$FRCNN_ROOT/models/VGG_CNN_M_1024`, it mainly concerns with the number of classes you want to train. Let's assume that the number of classes is `C (do not forget to count the `background` class). Then you should 
  - Modify `num_classes` to `C`;
  - Modify `num_output` in the `cls_score` layer to `C`
  - Modify `num_output` in the `bbox_pred` layer to `4 * C`

### Training


#### RPN + Fast RCNN
In the directory **$FRCNN_ROOT**, run the following command in the shell.

```sh
time ./tools/train_faster_rcnn_alt_opt.py --gpu 0 --net_name VGG_CNN_M_1024 \
    --weights data/imagenet_models/VGG_CNN_M_1024.caffemodel --imdb try1_train --set TRAIN.SCALES [224]
```
or 
```sh
time ./tools/train_faster_rcnn_alt_opt.py --gpu 0 --net_name VGG_CNN_M_1024 \
    --weights data/imagenet_models/VGG_CNN_M_1024.caffemodel --imdb try1_train --cfg experiments/cfgs/faster_rcnn_alt_opt.yml
```
or for end2end (default iters is 40000)
```sh
time ./tools/train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.caffemodel --imdb try1_train --cfg experiments/cfgs/faster_rcnn_end2end.yml --iters 1000 
```
- Be careful with the **imdb** argument as it specifies the dataset you will train on. 
- **Empty annotation files are NOT OK**. 
- To change the number of iterations, go to tools/train_faster_rcnn_alt_opt.py and the function get_solvers

### Testing
```sh
time ./tools/test_net.py --gpu 0 --def models/VGG_CNN_M_1024/faster_rcnn_alt_opt/faster_rcnn_test.pt \
    --net output/faster_rcnn_alt_opt/train/VGG_CNN_M_1024_faster_rcnn_final.caffemodel --imdb try1_test --cfg experiments/cfgs/faster_rcnn_alt_opt.yml
```
or for end2end
```sh
time ./tools/test_net.py --gpu 0 --def models/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt --net output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_iter_1000.caffemodel --imdb try1_test --cfg experiments/cfgs/faster_rcnn_end2end.yml
```



