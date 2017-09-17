# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:07:57 2017

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

# copy rows to df for which pictures are available
for s in shop_id:
    if int(s) in dataset.iloc[:,0].values.tolist():
        df.loc[len(df)] = dataset.iloc[dataset[dataset.shop_id == int(s)].index.tolist(),:].values.tolist()[0]

# converting float to int
df[['shop_id','picture_id']] = df[['shop_id','picture_id']].astype(int)

    
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
    

# transposing the test_df to match input dataset df
test_df = test_df.transpose()
test_df = test_df.reset_index(drop=True)

#new_df = df.iloc[:50, :]
# merging feature vector to input dataset df by column
final_df = pd.concat([test_df, df], axis=1)

# subsetting the data
x = final_df.iloc[:, :64].values
y = final_df.iloc[:, -1].values



############################# SVM ########################################
#SVM
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0, random_state = 0)
#
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
#x_test = sc_x.transform(x_test)
#
# Fitting SVM to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

############################ End SVM ######################################

# saving the model
from sklearn.externals import joblib
model_name = 'pizza.pkl'
joblib.dump(classifier, model_name)
scaling_model = 'scaling.pkl'
joblib.dump(sc_x, scaling_model)