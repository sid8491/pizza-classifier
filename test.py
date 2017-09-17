# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 20:50:52 2017

@author: Siddharth.Shukla01
"""

import pandas as pd
import numpy as np
import glob
from PIL import Image

# loading the model
from sklearn.externals import joblib
model_name = 'pizza.pkl'
scaling_model = 'scaling.pkl'
clf = joblib.load(model_name)
scaling = joblib.load(scaling_model)

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
#print (new_y_pred)

result = list(zip(test_files, new_y_pred))
print ('Result ------------> ', result)