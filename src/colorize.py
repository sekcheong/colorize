import os
import cv2
import numpy as np
import scipy as sp
import time
import warnings


from keras.utils import plot_model

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


from os.path                    import join, basename
from scipy.misc                 import imresize, imread
from datetime                   import datetime
from keras.models               import Sequential
from keras.layers               import Dense, Activation, Flatten, Dropout
from keras.layers.pooling       import MaxPooling2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.optimizers           import Adam
from keras.layers.advanced_activations import LeakyReLU



def imageLoad(path):
	return cv2.imread(path)


def imageIsGrayScale(img):
	if not np.array_equal(img[:,:,0],img[:,:,1]): return False 
	if not np.array_equal(img[:,:,1],img[:,:,2]): return False 
	return True


def imageRGB2YUV(img):
	out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	Y = out[:,:,0]
	U = out[:,:,1]
	V = out[:,:,2]
	return (Y, U, V)


def imageYUV2RGB(Y, U, V):
	if (U.shape != Y.shape):
		U = cv2.resize(U, (Y.shape[1], Y.shape[0]), interpolation = cv2.INTER_CUBIC)
	if (V.shape != Y.shape):
		V = cv2.resize(V, (Y.shape[1], Y.shape[0]), interpolation = cv2.INTER_CUBIC)
	yuv = (np.dstack([Y,U,V])).astype(np.uint8)
	img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
	return img


def imageCompressColor(Y, U, V, ratio=0.5):
	U = cv2.resize(U, (int(U.shape[1]*ratio), int(U.shape[0]*ratio) ) , interpolation = cv2.INTER_CUBIC)
	V = cv2.resize(V, (int(V.shape[1]*ratio), int(V.shape[0]*ratio) ) , interpolation = cv2.INTER_CUBIC)
	return (Y, U, V)


def imageDecompressColor(Y, U, V):
	U = cv2.resize(U, (Y.shape[1], Y.shape[0]), interpolation = cv2.INTER_CUBIC)
	V = cv2.resize(V, (Y.shape[1], Y.shape[0]), interpolation = cv2.INTER_CUBIC)
	return (Y, U, V)


def imageResizeCrop(img, size=224):
	width = img.shape[1]
	height = img.shape[0]
	ratio =  height / width
	
	if (width>height):
		#if landscape mode
		img = cv2.resize(img, (int(size/ratio), size), interpolation = cv2.INTER_CUBIC)
	else:
		#if protrait mode
		img = cv2.resize(img, (size, int(size*ratio)), interpolation = cv2.INTER_CUBIC)
	#crop the image to size 
	img = img[0:size, 0:size]

	return img


def imageFromComponent(data):
	out = np.zeros((data.shape[0], data.shape[1], 3), dtype='uint8')
	out[:, :, 0] = data[:, :]
	out[:, :, 1] = data[:, :]
	out[:, :, 2] = data[:, :]
	return out;


def imageConcat(images):
	n = len(images)
	w = images[0].shape[1]
	h = images[0].shape[0]
	totalW = w * n
	newImage = np.zeros(shape=(h, totalW, 3), dtype=np.uint8)
	pos = 0
	for img in images:
		newImage[:h, pos:pos+w]=img
		pos = pos + w
	return newImage


def processImages(path, outPath):
	for fname in os.listdir(path):
		if fname.lower().endswith(".jpg") or fname.lower().endswith(".png"): 
			imgPath = os.path.join(path, fname)
			img = imageLoad(imgPath)
			if not imageIsGrayScale(img):					
				img = imageResizeCrop(img)
				name = fname.split(".")[0]
				imgPath = os.path.join(outPath, name) + ".jpg"
				print(imgPath)
				cv2.imwrite(imgPath, img)


def VGG16():
	




#main 

if __name__ == "__main__":

	processImages('../images/mirflickr', '../images/processed')
	
	img = imageLoad('../images/processed/im5.jpg')  #cv2.imread('../images/mirflickr/im4183.jpg')
	#img = imageResizeCrop(img)

	print ('Shape:', img.shape[0],  img.dtype, '\nGray Level:', imageIsGrayScale(img))

	(Y, U, V) = imageRGB2YUV(img)
	(Y, U, V) = imageCompressColor(Y, U, V, 0.15)

	#(Y, U, V) = (imageFromComponent(Y), imageFromComponent(U), imageFromComponent(V))
	print(U.shape, ",", V.shape)
	img2 = imageYUV2RGB(Y, U, V)
	cv2.imshow('image', imageConcat([img, img2]) )

	#cv2.imshow('image', imageConcat([img, Y, U, V]))
	k = cv2.waitKey(0) & 0xFF

	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
	elif k == ord('s'): # wait for 's' key to save and exit
	    cv2.imwrite('out.png',img)
	    cv2.destroyAllWindows()
