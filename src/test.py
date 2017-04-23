#!/bin/python
import cv2
import numpy as np

def rgbtoYUC(img):
	return img

img = cv2.imread('im4183.jpg')


img = rgbtoYUC(img)

print ('img:', img.shape)
img_out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

print ('img_out:', img_out.shape)

cv2.imshow('im4183.jpg',img_out)

k = cv2.waitKey(0) & 0xFF

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
#elif k == ord('s'): # wait for 's' key to save and exit
#    cv2.imwrite('messigray.png',img)
#    cv2.destroyAllWindows()
