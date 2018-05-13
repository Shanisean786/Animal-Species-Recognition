#from global_l import fixed_size,bins,fd_hu_moments,fd_haralick,fd_histogram
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
import mahotas
##creating all machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=9)))

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

# import the feature vector and trained labels
h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print ("features shape: {}",global_features.shape)
print (" labels shape: {}",global_labels.shape)

print ("training started...")



# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),np.array(global_labels),test_size=0.1,random_state=9)
                                                                    
print("SPlitted Train and test data")
print("train data:",trainDataGlobal.shape)
print("test data:",testDataGlobal.shape)
print("train labels:",trainLabelsGlobal.shape)
print("testLabelsGlobal:",testLabelsGlobal.shape)
"""
# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()                                                                                        
"""

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


# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=100, random_state=9)
# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

###Saving the trained model
from sklearn.externals import joblib
joblib.dump(clf,'model.pkl')


# path to test data
test_path = "animal_database/test"
train_path="animal_database/train"
train_labels=os.listdir(train_path)

# loop through the test images
for file in os.listdir(test_path):
    # read the image
    file_1=test_path+"/"+file
    print(file_1)
    image = cv2.imread(file_1)
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    # resize the image
    image = cv2.resize(image, fixed_size)

  
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    print(global_features)
    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
                                                                                  
