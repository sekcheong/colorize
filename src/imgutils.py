import os
import cv2
import numpy as np
import scipy as sp
import random
from shutil import copyfile

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
	c = 1
	for fname in os.listdir(path):
		if fname.lower().endswith(".jpg") or fname.lower().endswith(".png") or fname.lower().endswith(".jpeg"): 
			imgPath = os.path.join(path, fname)
			img = imageLoad(imgPath)
			if not imageIsGrayScale(img):					
				img = imageResizeCrop(img)
				name = 'im{num:08d}'.format(num=c)
				imgPath = os.path.join(outPath, name) + ".jpg"
				print(imgPath)
				cv2.imwrite(imgPath, img)
				c += 1


def prepareDataSet(path, trainPath, tunePath, testPath, ratio = (0.7, 0.2, 0.1), size=10000):
	files = []
	for fname in os.listdir(path):
		if fname.lower().endswith(".jpg") or fname.lower().endswith(".png") or fname.lower().endswith(".jpeg"):
			files.append(fname)
	
	random.seed(838*838)
	random.shuffle(files)

	trainsize = int(size * ratio[0])
	tunesize = int(size * ratio[1])
	testsize = int(size - trainsize - tunesize)

	print(trainsize,",",tunesize,",",testsize)

	train = []
	tune = []
	test = []

	for i in range(trainsize):
		train.append(files.pop())

	for i in range(tunesize):
		tune.append(files.pop())

	for i in range(testsize):
		test.append(files.pop())

	for fname in train:
		src = os.path.join(path, fname)
		dest = os.path.join(trainPath, fname)
		copyfile(src, dest)

	for fname in tune:
		src = os.path.join(path, fname)
		dest = os.path.join(tunePath, fname)
		copyfile(src, dest)

	for fname in test:
		src = os.path.join(path, fname)
		dest = os.path.join(testPath, fname)
		copyfile(src, dest)



#main 

if __name__ == "__main__":

	#processImages('../images/mirflickr', '../images/processed')
	#prepareDataSet('../images/processed','../images/train','../images/tune','../images/test')
	
	img = imageLoad('../images/processed/im10110.jpg')
	#img = imageResizeCrop(img)

	# print ('Shape:', img.shape[0],  img.dtype, '\nGray Level:', imageIsGrayScale(img))

	(Y, U, V) = imageRGB2YUV(img)
	(Y, U, V) = imageCompressColor(Y, U, V, 0.1)

	#(Y, U, V) = (imageFromComponent(Y), imageFromComponent(U), imageFromComponent(V))
	


	# print(U.shape, ",", V.shape)
	img2 = imageYUV2RGB(Y, U, V)
	# cv2.imshow('image', imageConcat([img, img2]) )

	#img = imageConcat([img, Y, U, V])

	img = imageConcat([img, img2])

	cv2.imshow('image', img)
	k = cv2.waitKey(0) & 0xFF
	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
	elif k == ord('s'): # wait for 's' key to save and exit
		cv2.imwrite('out.png',img)
		cv2.destroyAllWindows()


