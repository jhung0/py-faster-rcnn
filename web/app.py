from flask import Flask, render_template, request
from flask_uploads import UploadSet, IMAGES, configure_uploads
import os
app = Flask(__name__, instance_path='/home/ubuntu/py-faster-rcnn/web/instance')
#app.config.from_object('app.default_settings')
#app.config.from_pyfile('app.cfg', silent=True) 

# Configure the image uploading via Flask-Uploads
images = UploadSet('images', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = '/home/ubuntu/images'
configure_uploads(app, images)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'image' in request.files:
        filename = images.save(request.files['image'])
        rec = Image(filename=filename)
        rec.store()
        return redirect(url_for('show', id=rec.id))
    return render_template('upload.html')

@app.route('/')
def showResult():
    return 'Flask Dockerized'


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
