"""
Strategy:
 - Use the upper right hand corner to calibrate coordinates
 - (?) Find coordinates somehow
 - For each predicted grid intersection, check (with a convolution) whether 
 	it's empty, black, or white.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import random

np.set_printoptions(threshold=np.nan)

def read(fname, thres = 190):
	img = cv2.imread(fname, 0)
	img = (255 * (img > thres)).astype(np.uint8)
	img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	return img
	# Use 0 for grayscale

def show(img, fname=""):
	if type(img) == type([]):
		for (x,f) in zip(img, fname):
			cv2.imshow(f, x)
	else:
		cv2.imshow(fname, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def findCircles(img,param2=16):
	# Cutoff seems to be param2 = 15.5. Higher = missing black stones. Lower = false pos.
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	h, w = img.shape
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=w/20.,\
					param1=100, param2=param2, minRadius=int(w/40.), maxRadius=int(w/20.))
	return circles

def drawCircles(img, circles):
	if circles is not None:
		for x in circles[0, :]:
			center = (x[0], x[1])
			radius = x[2]
			cv2.circle(img, center, radius, (0, 0, 200))
	return img


#circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
#                            param1=50,param2=30,minRadius=0,maxRadius=0)

if __name__ == "__main__":
	indices = [random.randint(1,861)]
	indices = [24]
	prefix = "tsumego/tsumego"
	suffix = ".png"
	
	for j in range(10):
		fnames=[prefix + str(i) + suffix for i in indices]
		imgs = [read(fname) for fname in fnames]
		allCircles = [findCircles(img) for img in imgs]	
		
		drawn = [drawCircles(img, circles) for (img, circles) in zip(imgs, allCircles)]
		show(drawn, fnames)
		indices[0] = random.randint(1,861)
		break

"""
7/11/18: findCircles works. param2 needs tuning.
	If > 15.5, then it'll miss black stones. If < 15.5, then it'll have false positives.
	Strategy: check once for two different param values. If the interior is mostly black,
		Then, pass it off. Otherwise, it better show up for both param values.
"""







