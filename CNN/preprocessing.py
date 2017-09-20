"""
import pandas as pd
import re
import os
df = pd.read_csv("df.csv")
df_set = df.loc[df['topping_code'] == 'S',]
files_set = list(df_set['shop_id'])


# this code will transform the sausage pizza images and make copies of them
# to balance the dataset
# as originally their were very few sausage images avaialable
import glob
all_name = glob.glob("D://CNN//Images_all//*.jpg")
all_names = []

for f in all_name:
	name = f.replace('\\','//')
	all_names.append(name)

#print all_names
source = "D://CNN//Images_all//"
destination = 'D://CNN//lowsample//'

finl_list = []
i = 0
for xx in all_names:
	reg = re.findall('shop_[\d]*', xx, re.I)
	reg = re.sub('shop_', '', reg[0])
#	print reg
	if int(reg) in files_set:
#		print 'true'
		finl_list.append(xx)
#print finl_list
#print files_set

for ss in finl_list:
	s = ss
	sd = s.replace('Images_all','lowsample')
	os.rename(ss, sd)

"""

import glob
all_name = glob.glob("D://CNN//lowsample//*.jpg")
all_names = []

for f in all_name:
	name = f.replace('\\','//')
	all_names.append(name)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


for nam in all_names:
	img = load_img(nam)
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)	
	
	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `lowsample/` directory
	i = 0
	for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='D://CNN//lowsample', save_prefix='sausage', save_format='jpg'):
		i += 1
		if i > 8:
			break  # otherwise the generator would loop indefinitely
		

