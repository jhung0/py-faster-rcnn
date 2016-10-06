import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import numpy as np
import subprocess
import pickle
from shutil import copyfile
import os
'''
Trains multiple times (at different learning rates)
'''
base_dir = '/home/ubuntu/py-faster-rcnn'
prototxt = "models/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt"
weights = "data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel"
cfg_file = "experiments/cfgs/faster_rcnn_end2end.yml"
output_dir = os.path.join(base_dir,'output/faster_rcnn_end2end/train/')
iterations = 100000
learning_range = [.01, .005, .0005, .0001]
for learning in learning_range:		
	print 'learning ', learning
        new_prototxt = "models/VGG_CNN_M_1024/faster_rcnn_end2end/solver_lr"+str(learning)+".prototxt"
        copyfile(prototxt, new_prototxt)
        with open(new_prototxt, "r") as f:
                lines = f.readlines()
        with open(new_prototxt, "w") as f:
                for line in lines:
                        if 'base_lr' not in line:
                                f.write(line)
                        else:
                                f.write('base_lr: '+str(learning))
        prototxt = new_prototxt
	proc = subprocess.Popen(['python', '/home/ubuntu/py-faster-rcnn/tools/train_net.py', '--gpu', '0', '--solver', prototxt, '--weights',  weights, '--imdb', 'try1_train', '--cfg', cfg_file, '--iters', str(iterations)], stdout=subprocess.PIPE)
	
	#save caffemodel as another name so it's not overwritten
	for caffemodel in os.listdir(output_dir):
		if '0000.caffemodel' in caffemodel:
			copyfile(os.path.join(output_dir, caffemodel),os.path.join(output_dir, caffemodel.split('.caffemodel')[0]+'_lr'+str(learning)+'.caffemodel')) 
