import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import numpy as np
import subprocess
import pickle
from shutil import copyfile
'''
Implements thresholdout algorithm to get generalization error results
'''
class Thresholdout():
	def __init__(self):
		self.error_threshold = 0.05
		self.tolerance = 0.01
		self.budget = 25 

		#Test on training data and test data
		self.prototxt = "models/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt"
		self.net = "output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_iter_40000.caffemodel"
		self.cfg_file = "experiments/cfgs/faster_rcnn_end2end.yml"

	def processOutput(self, proc):
		results = []
		for line in proc.stdout:
			if '_r_p_ap.pkl' in line:
				print line
				results_cls = pickle.load(open(line.strip()))
				results.append(results_cls['ap'])
		return results

	def checkBudget(self):
		if self.budget < 0:
			raise Exception('Budget ran out! END')

	def getHoldoutScore(self, holdout_score, train_score, test_score):
		if not train_score and not test_score:
                	raise Exception('nan')
        	if abs(train_score - test_score) < self.error_threshold + np.random.normal(0, self.tolerance):
                	holdout_score.append(train_score)
        	else:
                	holdout_score.append(test_score + np.random.normal(0, self.tolerance))
                	self.budget = self.budget - 1
		return holdout_score

	def getScores(self):
		proc = subprocess.Popen(['python', '/home/ubuntu/py-faster-rcnn/tools/test_net.py', '--gpu', '0', '--def', self.prototxt, '--net',  self.net, '--imdb', 'try1_trainfull', '--cfg', self.cfg_file], stdout=subprocess.PIPE)
        	results = self.processOutput(proc) 
        	train_score = np.mean(results)
        	proc = subprocess.Popen(['python', '/home/ubuntu/py-faster-rcnn/tools/test_net.py', '--gpu', '0', '--def', self.prototxt, '--net',  self.net, '--imdb', 'try1_test', '--cfg', self.cfg_file], stdout=subprocess.PIPE)
        	results = self.processOutput(proc)
        	test_score = np.mean(results)
		return train_score, test_score
	
	def oneCheck(self, holdout_score):
		self.checkBudget()
                train_score, test_score = self.getScores()
                print train_score, test_score
                holdout_score = self.getHoldoutScore(holdout_score, budget, train_score, test_score)
		return holdout_score

	def main(self):
		iter_range = range(10000, 100000, 10000)
        	learning_range = [.001]#[.01, .005, .001, .0005, .0001]
		holdout_score = []
        	for iterations in iter_range:
			print 'iteration ', iterations
                	self.net = "output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_iter_" + str(iterations) + ".caffemodel"
                	for learning in learning_range:
				'''
				print 'learning ', learning
                                new_prototxt = "models/VGG_CNN_M_1024/faster_rcnn_end2end/test_lr"+str(learning)+".prototxt"
                                copyfile(self.prototxt, new_prototxt)
                                with open(new_prototxt, "r") as f:
                                        lines = f.readlines()
                                with open(new_prototxt, "w") as f:
                                        for line in lines:
                                                if 'base_lr' not in line:
                                                        f.write(line)
                                                else:
                                                        f.write('base_lr: '+str(learning))
                                self.prototxt = new_prototxt
				'''
                                holdout_score = self.oneCheck(holdout_score)
                                print holdout_score
                                with open(self.name, 'a+') as f:
                                        print str(iterations)+","+str(learning)+","+holdout_score		
if __name__ == "__main__":
	thresholdout = Thresholdout()
	thresholdout.main()
