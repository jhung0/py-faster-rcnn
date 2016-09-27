import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import numpy as np
import subprocess
import pickle
'''
Implements thresholdout algorithm to get generalization error results
'''

error_threshold = 0.05
tolerance = 0.01
budget = 25 

#Test on training data and test data
prototxt = "models/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt"
cfg_file = "experiments/cfgs/faster_rcnn_end2end.yml"

for iterations in range(100000, 200000, 100000):
	if budget < 0:
		print 'Budget ran out! END'
		break
	net = "output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_iter_" + str(iterations) + ".caffemodel"
	proc = subprocess.Popen(['ls', '-l'], stdout=subprocess.PIPE)
	for line in proc.stdout:
		print line
	results = []
	proc = subprocess.Popen(['python', '/home/ubuntu/py-faster-rcnn/tools/test_net.py', '--gpu', '0', '--def', prototxt, '--net',  net, '--imdb', 'try1_trainfull', '--cfg', cfg_file], stdout=subprocess.PIPE)
	for line in proc.stdout:
		print line
		if '_r_p_ap.pkl' in line:
			print line
			results_cls = pickle.load(open(line.strip()))
			print results_cls
			results.append(results_cls['ap'])
	print results
	train_score = np.mean(results)
	print train_score
	#subprocess.call(["python", "/home/ubuntu/py-faster-rcnn/tools/test_net.py", '--gpu', '0', '--def', prototxt, '--net',  net, '--imdb', 'try1_trainfull', '--cfg', cfg_file]) 
	#subprocess.call(["python", "/home/ubuntu/py-faster-rcnn/tools/test_net.py", '--gpu', '0', '--def', prototxt, '--net',  net, '--imdb', 'try1_test', '--cfg', cfg_file]) 
	'''
	test_score
	if abs(train_score - test_score) < error_threshold + np.random.normal(0, tolerance):
		holdout_score = train_score
	else:
		holdout_score = test_score + np.random.normal(0, tolerance)
		budget = budget - 1 
	'''
