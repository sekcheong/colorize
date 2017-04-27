#!/bin/python
import os
import cv2
import numpy as np


def imageLoad(path):
	return cv2.imread(path)


def imageIsGrayScale(img):
	if not np.array_equal(img[:,:,0],img[:,:,1]): return False 
	if not np.array_equal(img[:,:,1],img[:,:,2]): return False 
	return True


def imageCompressColor(img):
	#convert image to YUV
	out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	Y = out[:,:,0]
	U = out[:,:,1]
	V = out[:,:,2]
	return (Y, U, V) 


def imageResizeCrop(img, size=244):
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
				print(imgPath)						
				img = imageResizeCrop(img)
				name = fname.split(".")[0]
				imgPath = os.path.join(outPath, name) + ".jpg"
				print(imgPath)
				cv2.imwrite(imgPath, img)



#main 

if __name__ == "__main__":


	processImages('../images/mirflickr', '../images/processed')
	img = imageLoad('../images/mirflickr/im3.jpg')  #cv2.imread('../images/mirflickr/im4183.jpg')
	img = imageResizeCrop(img)

	print ('Shape:', img.shape[0],  img.dtype, '\nGray Level:', imageIsGrayScale(img))

	(Y, U, V) = imageCompressColor(img)

	(Y, U, V) = (imageFromComponent(Y), imageFromComponent(U), imageFromComponent(V))


	cv2.imshow('image', imageConcat([img, Y, U, V]))
	k = cv2.waitKey(0) & 0xFF

	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
	elif k == ord('s'): # wait for 's' key to save and exit
	    cv2.imwrite('out.png',img)
	    cv2.destroyAllWindows()
