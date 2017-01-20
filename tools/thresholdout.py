import _init_paths
import matplotlib
matplotlib.use('Agg')
import caffe
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import numpy as np
import subprocess
import pickle
from shutil import copyfile
import os 
import argparse
import pprint
from fast_rcnn.test import test_net
from cStringIO import StringIO
import sys
'''
Implements thresholdout algorithm to get generalization error results
'''
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

class Thresholdout():
	def __init__(self):
		imdb = get_imdb('try1_test').gt_roidb()
		count = 0
		for i in imdb:
    			count += len(i['gt_classes'])
		self.n = count #holdout size
		self.error_threshold = 4*1.0/np.sqrt(self.n) #0.05
		self.tolerance = 1.0/np.sqrt(self.n) #0.01 #noise
		#self.budget = self.n*1.0/2*self.tolerance**2 #30 

		#Test on training data and test data
		self.prototxt = "models/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt"
		self.net = "output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_iter_40000.caffemodel"
		self.cfg_file = "experiments/cfgs/faster_rcnn_end2end.yml"
		self.name = '/home/ubuntu/try1/results/thresholdout.txt'

	def parse_args(self):
		"""
    		Parse input arguments
    		"""
    		parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    		parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    		parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    		parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    		parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    		parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    		parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    		parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    		parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    		if len(sys.argv) == 1:
        		parser.print_help()
        		sys.exit(1)

    		args = parser.parse_args()
    		return args

	def runTestNet(self, test_name):
		'''
		args = parse_args()

		print('Called with args:')
    		print(args)

    		if args.cfg_file is not None:
        		cfg_from_file(args.cfg_file)
    		if args.set_cfgs is not None:
        		cfg_from_list(args.set_cfgs)
		'''
		cfg_from_file(self.cfg_file)

    		print('Using config:')
    		pprint.pprint(cfg)

    		caffe.set_mode_gpu()
    		caffe.set_device(0)
    		net = caffe.Net(self.prototxt, self.net, caffe.TEST)
    		print 'prototxt ', self.prototxt
    		print 'caffemodel ', self.net
    		print 'test ', caffe.TEST
    		net.name = os.path.splitext(os.path.basename(self.net))[0]

    		imdb = get_imdb(test_name)
    		if not cfg.TEST.HAS_RPN:
        		imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    		test_net(net, imdb)

	def processOutput(self, output):
		results = []
		for line in output: #proc.stdout:
			print line
			if '_r_p_ap.pkl' in line:
				#print line
				results_cls = pickle.load(open(line.strip()))
				results.append(results_cls['ap'])
		return results

	def checkBudget(self):
		if self.budget < 0:
			print 'Budget ran out! END'
			return 0

	def getHoldoutScore(self, holdout_score, train_score, test_score):
		if not train_score and not test_score:
                	raise Exception('nan')
        	if abs(train_score - test_score) < self.error_threshold + np.random.normal(0, self.tolerance):
                	holdout_score.append(train_score)
        	else:
                	holdout_score.append(test_score + np.random.normal(0, self.tolerance))
                	#self.budget = self.budget - 1
		return holdout_score

	def getScores(self):
		#subprocess.call(['python', '/home/ubuntu/py-faster-rcnn/tools/test_net.py', '--gpu', '0', '--def', self.prototxt, '--net',  self.net, '--imdb', 'try1_trainfull', '--cfg', self.cfg_file])
		proc = subprocess.Popen(['python', '/home/ubuntu/py-faster-rcnn/tools/test_net.py', '--gpu', '0', '--def', self.prototxt, '--net',  self.net, '--imdb', 'try1_test', '--cfg', self.cfg_file], stdout=subprocess.PIPE)
		#output, errors = proc.communicate()
		#proc.wait()
		#proc = os.popen('python /home/ubuntu/py-faster-rcnn/tools/test_net.py --gpu 0 --def ' + self.prototxt + ' --net ' + self.net +' --imdb try1_trainfull --cfg '+self.cfg_file, 'r')
		#with Capturing() as output:
		#	self.runTestNet('try1_test')
		results = self.processOutput(proc.stdout)
		print 'results ', results
		test_score = np.mean(results)
        	proc = subprocess.Popen(['python', '/home/ubuntu/py-faster-rcnn/tools/test_net.py', '--gpu', '0', '--def', self.prototxt, '--net',  self.net, '--imdb', 'try1_trainfull', '--cfg', self.cfg_file], stdout=subprocess.PIPE)
        	results = self.processOutput(proc.stdout)
        	train_score = np.mean(results)
		return train_score, test_score
	
	def oneCheck(self, holdout_score):
                train_score, test_score = self.getScores()
                print train_score, test_score
                holdout_score = self.getHoldoutScore(holdout_score, train_score, test_score)
		return train_score, test_score, holdout_score

	def main(self):
		iter_range = range(10000, 100001, 10000)
        	learning_range = [.01, .001, .0001]
		holdout_score = []
		print self.n, self.error_threshold, self.tolerance
		for learning in learning_range:
			print 'learning ', learning
        		for iterations in iter_range:
				print 'iteration ', iterations
				self.net = "output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_lr" + str(learning) + "_iter_" + str(iterations) + ".caffemodel"
				#if self.checkBudget() == 0:
				#	return
				print self.net, self.prototxt
				train_score, test_score, holdout_score = self.oneCheck(holdout_score)
                                print 'holdout score', holdout_score
                                with open(self.name, 'a+') as f:
                                        line = str(iterations)+","+str(learning)+","+str(train_score)+","+str(test_score)+","+str(holdout_score[-1])
					print line
					f.write(line+'\n')	
if __name__ == "__main__":
	thresholdout = Thresholdout()
	thresholdout.main()
