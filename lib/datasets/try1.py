# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
#import datasets.pascal_voc
import datasets.try1
import os
import datasets.imdb
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
from operator import itemgetter

class try1(datasets.imdb):
    def __init__(self, image_set, devkit_path):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set #usually train or test
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'rbc', 'ring', 'gam', 'uncertain')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.tif']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.rpn_roidb #self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images',
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_try1_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_try1_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        with open(filename) as f:
            data = f.readlines() #each row is an element in a list
        num_objs = len(data)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

	difficult = np.zeros((num_objs), dtype=np.int32)
        # Load object bounding boxes into a data frame.
        for ix in range(num_objs):
            try:
                x1, y1, x2, y2, cls, df = data[ix].strip().split(' ')
            except:
                raise Exception('Error in reading data, line %s:%s'%(str(ix+1), data[ix]))
            #pixel indexes 0-based
            cls = self._class_to_ind[cls]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
	    difficult[ix] = df == 'True' 

        overlaps = scipy.sparse.csr_matrix(overlaps)
	
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
		'difficult' : difficult}

    def _write_try1_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', self.name, str(os.getpid()))#, comp_id)
	
	results = []
        for cls_ind, cls in enumerate(self.classes):
	    results.append({})
	    results[cls_ind]['boxes'] = []
	    results[cls_ind]['class_probs'] = []
	    results[cls_ind]['index'] = []
	    results[cls_ind]['class'] = cls
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = path + '_det_' + self._image_set + '_' + cls + '.txt'
	    print filename
            with open(filename, 'wt+') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
			results[cls_ind]['index'].append(index)
			results[cls_ind]['boxes'].append(dets[k, 0:4].tolist())
			results[cls_ind]['class_probs'].append(dets[k, -1])
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] , dets[k, 1] ,
                                       dets[k, 2] , dets[k, 3] ))
	'''
	#proposals per image
	results2 = []
	filename = path + '_det_' + self._image_set + '.pkl'
	for im_ind, index in enumerate(self.image_index):
	    results2.append({})
	    results2[im_ind]['boxes'] = {} #[]
	    #results2[im_ind]['classes'] = []
	    #results2[im_ind]['class_probs'] = []
	    results2[im_ind]['index'] = index
	    #filename = path + '_det_' + self._image_set + '_' + index + '.pkl'
	    #print filename
	    #with open(filename, 'wt+') as f:
	    for cls_ind, cls in enumerate(self.classes):
		    dets = all_boxes[cls_ind][im_ind]
		    if dets == []:
			continue
		    '''
		    for k in xrange(dets.shape[0]):
			f.write('{:s} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                                format(cls, dets[k, 0] , dets[k, 1],
                                       dets[k, 2] , dets[k, 3], dets[k, -1] ))
		    '''
		    num_detections = dets.shape[0]
		    print 'number of detections ', num_detections
		    for i in xrange(num_detections):
			box = tuple(dets[i, 0:4])
			if box not in results2[im_ind]['boxes']:
				results2[im_ind]['boxes'][box] = []
			results2[im_ind]['boxes'][box].append((cls, dets[i, -1]))  
		        #results2[im_ind]['classes'].append(cls)
		        #results2[im_ind]['class_probs'].append(dets[i, -1])
        with open(filename, 'wt+') as f:
		cPickle.dump(results2, f)
        '''
	return results

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def _do_python_eval(self, results, cls, output_dir='output', MINOVERLAP=0.5):
	rm_results = self.config['cleanup']
	recall = []
	prec = []
	ap = 0 #average precision?
	path = os.path.join(self._devkit_path, 'results', self.name, str(os.getpid()))
	gt = []
	tp = [] #true positives
	fp = [] #false positives
	
	results_CLS = results[map(itemgetter('class'), results).index(cls)]
	results_CLS_index = results_CLS['index']
	results_CLS_boxes = results_CLS['boxes']
	results_CLS_prob = results_CLS['class_probs']
	
	#sort detections by decreasing confidence
        sorted_ids = np.argsort(-np.array(results_CLS_prob))
        results_CLS_index = [results_CLS_index[i] for i in sorted_ids]
        results_CLS_boxes = [results_CLS_boxes[i] for i in sorted_ids]
        results_CLS_prob = [results_CLS_prob[i] for i in sorted_ids]

	npos = 0
	#extract ground truth	
	for i, index in enumerate(self.image_index):
	    gt.append(self._load_try1_annotation(index))
	    #filter out objects not in class
	    cls_indices = np.where(gt[i]['gt_classes'] == self.classes.index(cls))[0]
	    for key in gt[i]:
		try:
		    gt[i][key] = gt[i][key][cls_indices]
		except:
		    pass
	    gt[i]['index'] = index
	    gt[i]['det'] = np.zeros(len(gt[i]['gt_classes'])) 
	    npos = npos + sum(1 - gt[i]['difficult'])

	#for each detection  
	nd = len(results_CLS_index) #number of detections
	for nproposal in xrange(nd):
	    #print 'nproposal', nproposal
            _boxes = results_CLS_boxes[nproposal]
            _class_prob = results_CLS_prob[nproposal]
	    ov_max = -float("inf")
	    index = results_CLS_index[nproposal]
	    #assign detection to ground truth object if any
	    gt_i = gt[map(itemgetter('index'), gt).index(index)]
	    #print 'index', index
	    #print 'gt_i', gt_i
	    #print _boxes
 	    for ngt in xrange(len(gt_i['gt_classes'])):
		#print 'ngt', ngt
		gt_boxes = gt_i['boxes'][ngt]
		#print gt_boxes
		iw = min(_boxes[2], gt_boxes[2]) - max(_boxes[0], gt_boxes[0]) + 1 
		ih = min(_boxes[3], gt_boxes[3]) - max(_boxes[1], gt_boxes[1]) + 1
		#print 'iw, ih', iw, ih
		if iw > 0 and ih > 0:
		    #compute overlap as area of intersection / area of union
		    ua = (_boxes[2] - _boxes[0] + 1)*(_boxes[3] - _boxes[1] + 1) + (gt_boxes[2] - gt_boxes[0] + 1)*(gt_boxes[3] - gt_boxes[1] + 1) - iw*ih
		    ov = iw*ih*1.0/ua
		    if ov > ov_max:
			ov_max = ov
			ngt_max = ngt		
	    #assign detection as true positive/don't care/false positive
	    if ov_max >= MINOVERLAP:
		#print ov_max
		if not gt_i['difficult'][ngt_max]: 
		    if not gt_i['det'][ngt_max]:
			tp.append(1)
			fp.append(0)
			gt_i['det'][ngt_max] = 1 #true positive
			#print gt_boxes
		    else:
			tp.append(0)
			fp.append(1) #false positive (multiple detection)
	    else:
		tp.append(0)
		fp.append(1)
	#compute precision and recall
	fp = np.cumsum(fp)
	tp = np.cumsum(tp)
	rec = tp*1.0/npos
	prec = tp*1.0/(fp+tp)

	#compute average precision
	for t in np.linspace(0, 1.0, endpoint=True, num=11):
	    try:
		p = np.max(prec[rec>=t])
	    except:
		p = 0
	    ap = ap + p*1.0/11	
	return rec, prec, ap, results_CLS_prob

    def calculate_auc(self, recall, prec):
	mrec = [0] + recall + [1]
	mrec = np.array(mrec)
	mpre = [0] + prec + [0]
	mpre = np.array(mpre)
	for i in np.arange(len(mpre)-2, -1, -1):
	    mpre[i] = max(mpre[i], mpre[i+1])
	i = np.where(mrec[1:] != mrec[:-1])[0]
	ap = sum((mrec[i+1] - mrec[i])*mpre[i+1])
	return ap

    def evaluate_detections(self, all_boxes, output_dir ):
        #MATLAB evaluation
	#comp_id = self._write_try1_results_file(all_boxes)
        #self._do_matlab_eval(comp_id, output_dir)
	
	#PYTHON evaluation
	recalls = []
	precs = []
	aps = []
	ap_aucs = []
	results = self._write_try1_results_file(all_boxes )
	for cls in self._classes:
	    if cls != '__background__':
		print cls
		recall, prec, ap, thresh = self._do_python_eval(results, cls, output_dir, 0.5)
	    	ap_auc = self.calculate_auc(recall, prec)
		recalls.append(recall)
		precs.append(prec)
		aps.append(ap)
		ap_aucs.append(ap_auc)
		print 'avg precision',ap, ap_auc
		with open(os.path.join(output_dir, str(os.getpid()) +'_det_'+ cls + '_r_p_ap.pkl'), 'w') as f:
            		cPickle.dump({'rec': recall, 'prec': prec, 'ap': ap, 'thresh':thresh}, f)
	
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.try1('train', '')
    res = d.roidb
    from IPython import embed; embed()
