from PIL import Image
import os
import numpy as np
path="C:/Users/Shani Sharma/Desktop/ML Projects/Animal Specie Recogniton/Data Sets/animal_database/";
Xlist=[]  #Stores the list of pixel values
Ylist=[] #Stores the list of Animal Species


#Creation of Training Set
for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        print(path+directory+"/"+file)
        img=Image.open(path+directory+"/"+file)
        featurevector=np.array(img).flatten()[:50]
        Xlist.append(featurevector)
        Ylist.append(directory)
		
#Splitting Training data into 80 % and rest 20% is test data
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xlist,Ylist,test_size=0.2,random_state=0) 

#Selecting Support Vector Machine Algorithm for training the model
from sklearn.svm import SVC
#classifier= SVC(kernel='linear',random_state=0)
#classifier=SVC(kernel='Rbf', random_state=0)
classifier.fit(xtrain,ytrain)

y_pred=classifier.predict(xtest)


#Calculating the accuracy
from sklearn.metrics import accuracy_score

print("Accuracy is :",accuracy_score(ytest,y_pred))