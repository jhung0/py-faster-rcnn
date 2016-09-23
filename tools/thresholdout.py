import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import numpy as np
import subprocess
'''
Implements thresholdout algorithm to get generalization error results
'''

error_threshold = 0.05
tolerance = 0.01
budget = 25 

#Test on training data and test data
prototxt = "models/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt"
cfg_file = "experiments/cfgs/faster_rcnn_end2end.yml"

for iterations in range(10000, 20000, 10000):
	if budget < 0:
		print 'Budget ran out! END'
		break
	net = "output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_iter_" + str(iterations) + ".caffemodel"

	set_args = "TRAIN.SCALES"
	subprocess.call(['python', '/home/ubuntu/py-faster-rcnn/tools/test_net.py', '--gpu', '0', '--def', prototxt, '--net',  net, '--imdb', 'try1_train', '--cfg', cfg_file, '--set', set_args])
	#subprocess.call(["python", "/home/ubuntu/py-faster-rcnn/tools/test_net.py", '--gpu', '0', '--def', prototxt, '--net',  net, '--imdb', 'try1_test', '--cfg', cfg_file]) 
	train_score
	test_score
	if abs(train_score - test_score) < error_threshold + np.random.normal(0, tolerance):
		holdout_score = train_score
	else:
		holdout_score = test_score + np.random.normal(0, tolerance)
		budget = budget - 1 
