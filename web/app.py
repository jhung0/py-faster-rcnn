from flask import Flask, redirect, url_for, render_template, request
from flask_uploads import UploadSet, IMAGES, configure_uploads
import os, sys
main_dir = '/home/ubuntu/py-faster-rcnn'
sys.path.insert(0, main_dir)
sys.path.insert(0, os.path.join(main_dir, 'lib'))
sys.path.insert(0, os.path.join(main_dir, 'caffe-fast-rcnn', 'python'))
import matplotlib
matplotlib.use('Agg')
import caffe
from run2stage import StageOne, StageTwo, CreateSvg, get_dimensions 
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
import uuid
import time

app = Flask(__name__,static_path='/home/ubuntu/py-faster-rcnn/web/static', instance_path='/home/ubuntu/py-faster-rcnn/web/instance')
#app.config.from_object('app.default_settings')
#app.config.from_pyfile('app.cfg', silent=True) 

# Configure the image uploading via Flask-Uploads
images = UploadSet('images', IMAGES)
app.config['UPLOADED_IMAGES_DEST'] = '/home/ubuntu/py-faster-rcnn/web/static/images'
svg_path = '/home/ubuntu/py-faster-rcnn/web/static/svg'
configure_uploads(app, images)

def predict(filename, caffemodel1, prototxt1, classes1, cfg_file1, caffemodel2, prototxt2, classes2, mean2, gpu_id=0, output_dir=svg_path):
    cfg_from_file(cfg_file1)
    cfg.GPU_ID = gpu_id

    while not os.path.exists(caffemodel1):
        print('Waiting for {} to exist...'.format(caffemodel1))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    
    dimensions = get_dimensions(filename)
    cfg_from_list(['TEST.SCALES', str([min(dimensions)]), 'TEST.MAX_SIZE', str(max(dimensions))])
    nms_dets = StageOne(filename, prototxt1, caffemodel1, classes1, THRESHOLD=1.0/len(classes1))
    stage2_probs = StageTwo(filename, prototxt2, caffemodel2, nms_dets[classes1.index('other')][0], classes2, mean2)
    return CreateSvg(output_dir, filename, nms_dets, stage2_probs, classes2)
	

@app.route("/predictions/<id>", methods=['GET'])
def show_prediction(id):
	#id = str(uuid.uuid3(uuid.NAMESPACE_DNS, id))
	return render_template("prediction.html", id=os.path.join(svg_path, id))

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'image' in request.files:
        filename = images.save(request.files['image'])
	caffemodel1 = os.path.join(main_dir, 'output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_lr0.01_iter_100000.caffemodel')	
	prototxt1 = os.path.join(main_dir, 'models/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt')
	classes1 = ['__background__', 'rbc', 'other']
	cfg_file1 = os.path.join(main_dir, 'experiments/cfgs/faster_rcnn_end2end.yml')
	caffemodel2 = os.path.join(main_dir, 'caffe-fast-rcnn/models/bvlc_reference_caffenet/caffenet_train_iter_40000.caffemodel')
	prototxt2 = os.path.join(main_dir, 'caffe-fast-rcnn/models/bvlc_reference_caffenet/deploy.prototxt')
	classes2 = ['__background__', 'rbc', 'trophozoite', 'schizont', 'ring', 'gametocyte', 'leukocyte']
	mean2 = [189.97, 133.83, 149.26]
	
	filename = os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename)	
	prediction_id = predict(filename, caffemodel1, prototxt1, classes1, cfg_file1, caffemodel2, prototxt2, classes2, mean2)
	prediction_id = os.path.relpath(prediction_id, svg_path)
	print prediction_id
        return redirect(url_for("show_prediction", id=prediction_id)) 

    return render_template('upload.html')#, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
