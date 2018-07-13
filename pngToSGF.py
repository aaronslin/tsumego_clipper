"""
Strategy:
 - Use the upper right hand corner to calibrate coordinates
 - (?) Find coordinates somehow
 - For each predicted grid intersection, check (with a convolution) whether 
 	it's empty, black, or white.
"""

import numpy as np
import cv2
import sys
import random

np.set_printoptions(threshold=np.nan)
GRAY_THRES = 190
BLACK_THRES = 10
WHITE_THRES = 255 - BLACK_THRES
MAX_IMG = 861
RED = (0,0,200)
BLUE = (200,0,0)
GREEN = (0,200,0)

BLACK_CHAR = "b"
WHITE_CHAR = "w"
EMPTY_CHAR = "."


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
	if type(fname) == type(0):
		fname = getFname(fname)
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

def drawCircles(img, circles, color=RED):
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

def intersectCircles(C1, C2, distThres):
	# dist: The maximum allowable distance between two centers to count as the same circle
	def circDist(c1, c2):
		# Distance between the centers
		dist = np.linalg.norm(c1[0:2]-c2[0:2])
		return dist
	
	# Definitely has room for optimization
	minDist = [min([circDist(c1, c2) for c2 in C2]) for c1 in C1]
	intersect = np.array([c1 for (c1, d) in zip(C1, minDist) if d < distThres])
	return intersect

def findWhiteStones_safe(img):
	# Safe because it misses some white stones
	loParam = 10
	hiParam = 16.9
	loCircles = findCircles(img, param2=loParam)[0]
	hiCircles = findCircles(img, param2=hiParam)[0]
	
	loAvgFill = [findAvgFill(img, circ) for circ in loCircles]
	loWhiteCircles = np.array([circ for (circ, fill) in zip(loCircles, loAvgFill) if fill > GRAY_THRES])

	# Within half a radius
	distThres = (img.shape[1])/(36. * 2)	
	intersect = intersectCircles(loWhiteCircles, hiCircles, distThres)
	return intersect

def generateImages(indices):
	# indices (int): generates (indices) random tsumego
	# indices (list): uses (indices) as the tsumego to look at
	if type(indices) == type(1):
		indices = [random.randint(1, MAX_IMG) for _ in range(indices)]
	imgs = [read(i) for i in indices]
	return imgs, indices

def _test_circleFunc(indices, func, **kwargs):
	print("Testing", func.__name__, ":", indices)
	imgs, indices = generateImages(indices)
	print(kwargs)
	circles = [func(img, **kwargs) for img in imgs]
	drawn = [drawCircles(img, circ) for (img, circ) in zip(imgs, circles)]
	for (pic,i) in zip(drawn,indices):
		show(pic, i)

def getTopRightCoord(img):
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	h,w = img.shape
	for x in range(w-1, -1, -1):
		for y in range(0, h, 1):
			if img[y][x] < BLACK_THRES:
				return (y,x)

def getGrid(img, blackStones):
	# Assumes that top right corner is empty
	(topY, rightX) = getTopRightCoord(img)
	# Assumes that the stone diam ~= grid length
	avgRad = np.mean(blackStones, axis=0)[2] 	
	leftmostBlack = min(blackStones, key=lambda x: x[0])
	hasFirstColStone = np.round(leftmostBlack[0]/avgRad) % 2 == 1
	# Assumes that the image is cropped to have 0 margin
	leftX = int(np.round(hasFirstColStone * avgRad))
	gridLen = (rightX - leftX)/18.0
	
	return (topY, leftX), gridLen

	# For testing purposes:
	for i in range(19):
		for j in range(19):
			x = int(leftX + i*gridLen)
			y = int(topY + j*gridLen)
			cv2.circle(img,(x,y), 2, GREEN)
	show(img)


def colorAtCoords(img, coords, gridLen):
	# Returns _ for no_stone, b for black, w for white
	patchRadius = gridLen/4.0
	h,w,_ = img.shape
	(y,x) = coords
	if y >= h or x >= w:
		return EMPTY_CHAR
	avgFill = findAvgFill(img, np.array([x, y, patchRadius]))
	if avgFill < BLACK_THRES:
		return BLACK_CHAR
	elif avgFill > WHITE_THRES:
		return WHITE_CHAR
	else:
		return EMPTY_CHAR

def getImageCoords(topleft, gridLen):
	(Y,X) = topleft
	coords = [(int(Y + i*gridLen), int(X + j*gridLen)) \
				for j in range(19) \
				for i in range(19)]
	return coords

def predictStones(i):
	img = read(i)
	return predictStonesByImg(img, False)

def predictStonesByImg(img, drawStones=True):
	blackStones = findBlackStones(img)
	topLeft, gridLen = getGrid(img, blackStones)
	imgCoords = getImageCoords(topLeft, gridLen)	
	predicted = "".join([colorAtCoords(img, c, gridLen) for c in imgCoords])
	if not drawStones:
		return predicted

	for (p,c) in zip(predicted, imgCoords):
		color = GREEN
		if p == EMPTY_CHAR:
			continue
		if p == BLACK_CHAR:
			color = BLUE
		if p == WHITE_CHAR:
			color = RED
		cv2.circle(img, (c[1], c[0]), 2, color)
	return predicted, img

def _test_predictStones(indices):
	imgs, indices = generateImages(indices)
	for (img,i) in zip(imgs,indices):
		predicted, drawn = predictStonesByImg(img)
		show(drawn, i)

if __name__ == "__main__":
	indices = 50
	_test_predictStones(indices)


"""
7/11/18: findCircles works. param2 needs tuning.
	If > 15.5, then it'll miss black stones. If < 15.5, then it'll have false positives.
	Strategy: check once for two different param values. If the interior is mostly black,
		Then, pass it off. Otherwise, it better show up for both param values.

7/12/18: findWhiteStones works with 95% accuracy.
	If param2 = 16.9, then there won't be any false positives.
	Unfortunately, there are white stones that will be missed because of the sharp threshold at 15.5.
	
	Strategy: Find the grid, given some confident stone locations. 
		Top right corner is always empty (maybe not in other packets though)
			Compute by finding min distance from top right of image.
			Update: looks like this is true for all three Cho Chikun packets
		Estimate the grid by using the radius of the stones
		Once given grid points, check whether there is a cross inside. 

7/12/18: predictStones() is the main function to use:
	returns a length 361 string with {., b, w} that can be parsed to become an SGF
	Todo next: parse the string -> SGF and then find a JS library for displaying SGF on a browser.

"""







