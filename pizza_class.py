# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 20:45:06 2017

@author: Siddharth.Shukla01
"""
# importing libraries
import pandas as pd
import numpy as np
import glob
import re
from PIL import Image


#getting dataset
dataset = pd.read_csv('data.csv', sep = ';')

#copying structure of original dataset
df = pd.DataFrame(data=None, columns = dataset.columns)


#get list of all files
files = glob.glob("Images_all/*.jpg")

shop_id = []

for f in files:
    shop = re.findall('shop_[\d]*', f, re.I)
    shop = re.sub('shop_', '', shop[0])
    shop_id.append(shop)
#    pic = re.findall('picture_[\d]*', f, re.I)
#    pic = re.sub('picture_', '', pic[0])

# copy rows to df for which pictures are available
for s in shop_id:
    if int(s) in dataset.iloc[:,0].values.tolist():
        df.loc[len(df)] = dataset.iloc[dataset[dataset.shop_id == int(s)].index.tolist(),:].values.tolist()[0]

# converting float to int
df[['shop_id','picture_id']] = df[['shop_id','picture_id']].astype(int)
#print (df.groupby('topping_code').count())

#for i in range(0,5):
#    print (files[i])
#    img = Image.open(files[i])
#    width, height = img.size
#    pixel_values = list(img.getdata())
#    pixel_values = np.array(pixel_values).reshape((width, height, 3))
#    #print (pixel_values[400][500][1])
#    red = 0
#    green = 0
#    blue = 0
#    for w in range(0,width):
#        for h in range(0,height):
#            r,g,b = img.getpixel((w,h))
##            r,g,b = rgb
#            red = red + r
#            green = green + g
#            blue = blue + b
#    
#    print (red/(width*height))
#    print (green/(width*height))
#    print (blue/(width*height))

    
test_df = pd.DataFrame()

# extracting features from image
# and taking window as 8
for i in range(len(df)):
    print (i)
    img = Image.open(files[i])
    blocks = 4
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in img.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    feature = pd.DataFrame(feature)
    test_df = pd.concat([test_df, feature], axis = 1)
#    print (test_df)
    

# transposing the test_df to match input dataset df
test_df = test_df.transpose()
test_df = test_df.reset_index(drop=True)


temp = pd.DataFrame(data=df, columns= df.columns, index=range(50))

# merging feature vector to input dataset df by column
#final_df = pd.concat([test_df, temp], axis=1)
final_df = pd.concat([test_df, df], axis=1)

# subsetting the data
x = final_df.iloc[:, :64].values
y = final_df.iloc[:, -1].values


############################# k-NN ###################################
'''
# splitting data into Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# fitting KNN to training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 1)
classifier.fit(x_train, y_train)

# Predicting Test set results
y_pred = classifier.predict(x_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculating accuracy and kappa
from sklearn.metrics import accuracy_score, cohen_kappa_score
accuracy_score(y_test, y_pred) * 100
cohen_kappa_score(y_test, y_pred) * 100
'''
############################# End k-NN ###################################

############################# SVM ########################################
#SVM
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
#
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#
# Fitting SVM to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)
#
# Predicting Test set results
y_pred = classifier.predict(x_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculating accuracy and kappa
from sklearn.metrics import accuracy_score, cohen_kappa_score
accuracy_score(y_test, y_pred) * 100
cohen_kappa_score(y_test, y_pred) * 100

############################ End SVM ######################################

# saving the model
from sklearn.externals import joblib
model_name = 'pizza.pkl'
joblib.dump(classifier, model_name)
scaling_model = 'scaling.pkl'
joblib.dump(sc_x, scaling_model)

# loading the model
from sklearn.externals import joblib
model_name = 'pizza.pkl'
scaling_model = 'scaling.pkl'
clf = joblib.load(model_name)
scaling = joblib.load(scaling_model)

#def test(folder = 'test/'):
test_files = glob.glob('test/' + "*.jpg")
new_test_df = pd.DataFrame()
for t in test_files:
    print (t)
    test_img = Image.open(t)
    print (test_img.size)
    blocks = 4
    test_feature = [0] * blocks * blocks * blocks
    for pixel in test_img.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        test_feature[idx] += 1
    test_feature = pd.DataFrame(test_feature)
    new_test_df = pd.concat([new_test_df, test_feature], axis = 1)
new_test_df = new_test_df.transpose()
new_test_df = new_test_df.reset_index(drop=True)
new_y_test = new_test_df.iloc[:, :].values

new_y_test = scaling.transform(new_y_test)
new_y_pred = clf.predict(new_y_test)
print (new_y_pred)




#test()

