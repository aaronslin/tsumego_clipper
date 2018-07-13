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
GRAY_THRES = 190
MAX_IMG = 861

def getFname(index):
	prefix = "tsumego/tsumego"
	suffix = ".png"
	return prefix + str(index) + suffix

def read(index):
	fname = getFname(index)
	img = cv2.imread(fname, 0)
	img = (255 * (img > GRAY_THRES)).astype(np.uint8)
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

def findAvgFill(img, circ):
	# Checks the average fill of the inscribed square	
	# Note: circ[0] is the x coordinate, which is along axis=1
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	r = circ[2] / np.sqrt(2)
	w1, w2 = int(circ[0]-r), int(circ[0]+r)+1
	h1, h2 = int(circ[1]-r), int(circ[1]+r)+1

	patch = img[h1:h2, w1:w2]
	return np.mean(patch)

def findBlackStones(img):
	param2 = 10			
	# Experimentally determined
	circles = findCircles(img,param2=param2)[0]
	avgFill = [findAvgFill(img, circ) for circ in circles]
	return np.array([circ for (circ, fill) in zip(circles, avgFill) if fill < GRAY_THRES])

def drawCircles(img, circles, color=(0,0,200)):
	if circles is None:
		assert("Error in drawCircles(): circles is None")
	elif type(circles) is not type(np.array([])):
		assert("Error in drawCircles(): circles is not type np.ndarray")
	elif len(circles.shape) < 2 or circles.shape[0] == 0:
		assert("Error in drawCircles(): circles is a degenerate array")
	else:
		if len(circles.shape) == 3 and circles.shape[0] == 1:
			circles = circles[0]
		for x in circles:
			center = (x[0], x[1])
			radius = x[2]
			cv2.circle(img, center, radius, color)
	return img

def generateImages(indices):
	# indices (int): generates (indices) random tsumego
	# indices (list): uses (indices) as the tsumego to look at
	if type(indices) == type(1):
		indices = [random.randint(1, MAX_IMG) for _ in range(indices)]
	imgs = [read(i) for i in indices]
	return imgs, indices

def _test_findCircleFunc(indices, func, **kwargs):
	imgs, indices = generateImages(indices)
	blackStones = [func(img, **kwargs) for img in imgs]
	drawn = [drawCircles(img, circ) for (img, circ) in zip(imgs, blackStones)]
	for (pic,i) in zip(drawn,indices):
		show(pic, getFname(i))
	print("Tested", func.__name__, ":" indices)

if __name__ == "__main__":
	fname = getFname(24)
	
	indices = [371, 601, 317, 451, 95]
	indices = 20
	_test_findCircleFunc(indices, findBlackStones)

	sys.exit(1)

"""
7/11/18: findCircles works. param2 needs tuning.
	If > 15.5, then it'll miss black stones. If < 15.5, then it'll have false positives.
	Strategy: check once for two different param values. If the interior is mostly black,
		Then, pass it off. Otherwise, it better show up for both param values.
"""







