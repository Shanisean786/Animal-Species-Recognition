import warnings
warnings.filterwarnings('ignore')
#############################
#from global_l import fixed_size,bins,fd_hu_moments,fd_haralick,fd_histogram,train_labels
import flask
import numpy as np
import mahotas
import cv2
from scipy import misc

#Image Processing
#from previous file
fixed_size=tuple((384,256))

#bins size
bins=8

def fd_hu_moments(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    feature=cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#feature-descriptor-2: haralick Texture
def fd_haralick(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #compute the haralick texture feature vector
    haralick=mahotas.features.haralick(gray).mean(axis=0)
    return haralick


#feature-descriptor-3: Color Histogram
def fd_histogram(image,mask=None):
    # convert the image to hsv color space
    image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #compute color histogram
    hist=cv2.calcHist([image],[0,1,2],None,[bins,bins,bins],[0,256,0,256,0,256])
    #normalise the histogram
    cv2.normalize(hist,hist)
    return hist.flatten()
import os

from sklearn.externals import joblib
from flask import Flask,render_template,request
train_labels=os.listdir("animal_database/train")
print(train_labels)
app=Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
	return flask.render_template('index.html')
@app.route('/predict',methods=['POST'])
def make_prediction():
	if request.method=='POST':
		file=request.files['image']
		if not file:
			return render_template('index.html',label="No file")
		image=misc.imread(file)
		image =cv2.resize(image, fixed_size)
		fv_hu_moments = fd_hu_moments(image)
		fv_haralick   = fd_haralick(image)
		fv_histogram  = fd_histogram(image)
		global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
		prediction=model.predict(global_feature.reshape(1,-1))[0]
		label=train_labels[prediction]
		return render_template('index.html',label=label,file=file)
		
if __name__=='__main__':
	model=joblib.load('model.pkl')
	app.run(host='0.0.0.0',port=8000,debug=True)

