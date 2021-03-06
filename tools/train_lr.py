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
weights = "data/imagenet_models/VGG_CNN_M_1024.caffemodel"
cfg_file = "experiments/cfgs/faster_rcnn_end2end.yml"
output_dir = os.path.join(base_dir,'output/faster_rcnn_end2end/train/')
iterations = 100000
learning_range = [.01, .001, .0001]#
for learning in learning_range:		
	print 'learning ', learning
        new_prototxt = "models/VGG_CNN_M_1024/faster_rcnn_end2end/solver_lr"+str(learning)+".prototxt"
        copyfile(prototxt, new_prototxt)
        with open(new_prototxt, "r") as f:
                lines = f.readlines()
        with open(new_prototxt, "w") as f:
                for line in lines:
                        if 'base_lr' in line:
				f.write('base_lr: '+str(learning)+"\n")
                        elif 'snapshot_prefix' in line:
				f.write('snapshot_prefix: "vgg_cnn_m_1024_faster_rcnn_lr'+str(learning)+'"')
			else:
				f.write(line)
        prototxt = new_prototxt
	#subprocess.call('python /home/ubuntu/py-faster-rcnn/tools/train_net.py --gpu 0 --solver '+ prototxt +' --weights ' +  weights +' --imdb try1_train --cfg '+ cfg_file+ ' --iters '+ str(iterations) + ' &', shell=True)
	subprocess.Popen(['python', '/home/ubuntu/py-faster-rcnn/tools/train_net.py', '--gpu', '0', '--solver', prototxt, '--weights',  weights, '--imdb', 'try1_train', '--cfg', cfg_file, '--iters', str(iterations)])
	'''
	try:
		subprocess.call(['python', '/home/ubuntu/py-faster-rcnn/tools/train_net.py', '--gpu', '0', '--solver', prototxt, '--weights',  weights, '--imdb', 'try1_train', '--cfg', cfg_file, '--iters', str(iterations)])
	except:
		print 'learning rate ' + learning + 'failed'
		continue
	
	#save caffemodel as another name so it's not overwritten
	for caffemodel in os.listdir(output_dir):
		if '0000.caffemodel' in caffemodel:
			copyfile(os.path.join(output_dir, caffemodel),os.path.join(output_dir, caffemodel.split('.caffemodel')[0]+'_lr'+str(learning)+'.caffemodel'))
	''' 
