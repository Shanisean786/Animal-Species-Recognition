import warnings
warnings.filterwarnings('ignore')

#organise imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

#fixed-sizes for image
fixed_size=tuple((200,200))

#patch to taining data
train_path="animal_database/train"

#no. of trees for random Forests
num_trees=100

#bins for histogram
bins=8

#train_test_split
test_size=0.10

#seed for reproducing same results
seed=9

#feature-descriptor-1: Hu Moments
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


#get the training labels
train_labels=os.listdir(train_path)

#sort the training labels
train_labels.sort()
#print(train_labels)



#empty list to hold feature vectors and labels
global_features=[]
labels=[]
i,j=0,0
k=0
#print(train_labels)
"""
dir=os.path.join(train_path,train_labels[0])
image=cv2.imread(dir+'/001.jpg')
fv_hu_moments=fd_histogram(image)
print(fv_hu_moments)""" 
for training_name in train_labels:
    dir=os.path.join(train_path,training_name)
    current_label=training_name
    #print (dir)
    k=1
    for x in os.listdir(dir):
        file=dir+"/"+x
        #print(file)
        image=cv2.imread(file)
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
		
        #break
        image=cv2.resize(image,fixed_size)

        ###Global Feature Extraction###
        fv_hu_moments=fd_hu_moments(image)
        fv_haralick=fd_haralick(image)
        fv_histogram=fd_histogram(image)

        #concatenate features
        global_feature=np.hstack([fv_histogram,fv_haralick,fv_hu_moments])

        #update lists
        labels.append(current_label)
        global_features.append(global_feature)

        i+=1
        k+=1
    print("Status Processed Folder: ",current_label)
    j +=1
print ("Status: completed Global Extraction")


#Overall feature vector size
print("Feature vector size is: ",np.array(global_features).shape)

#overall training label size
print("Training Labels: ",np.array(labels).shape)

#encode the target labels
targetNames = np.unique(labels)
le=LabelEncoder()
target=le.fit_transform(labels)
##print("Status: training labels Encoded")

#normalize the feature vector in the range (0,1)
scaler=MinMaxScaler(feature_range=(0,1))
rescaled_features=scaler.fit_transform(global_features)
##print("Status: feature vector normalize")

##print("Target variables",target)
##print("target variables shape: ",np.array(target).shape)

##Saving the Features Extracted into HDF5 File

h5f_data=h5py.File('output/data.h5','w')
h5f_data.create_dataset('dataset_1',data=np.array(rescaled_features))

h5f_label=h5py.File('output/labels.h5','w')
h5f_label.create_dataset('dataset_1',data=np.array(target))

h5f_data.close()
h5f_label.close()
print("End of training")
















    



