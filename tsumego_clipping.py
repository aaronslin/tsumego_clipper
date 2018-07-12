"""
Strategy: In the PDF, set an upper bound on "allowable whitespace"
	Automatically split by these divisions, and then trim the surrounding whitespace.
"""

from scipy import misc
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

pdfname = "cho-2-intermediate-mobile.pdf"
def pdfToPages(pdfname):
	from pdf2image import convert_from_path

	pages = convert_from_path(pdfname)
	for (n,page) in enumerate(pages):
		fname = "pages/page"+str(n)+".png"
		page.save(fname, "PNG")

def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def notWhitespace(img, ax=1):
	whiteThres = 254.0
	return np.logical_not(np.all(img>whiteThres, axis=ax))

def cropOne(img):
	# mask: a 1D array of True/False values
	# margin: <= 7, where 7 is appears to be the number of whitespace rows
	# between the tsumego itself and the problem label

	mask = notWhitespace(img)
	splits = []
	indices = []
	prev = False
	for (i,r) in enumerate(mask):
		if prev == True and r == False:
			splits.append(np.array(indices))
			indices = []
		elif r == False:
			pass
		elif r == True:
			indices.append(i)			
		prev = r

	return [img[s] for s in splits]

def cropVert(imgs):
	# input: array of imgs, not a mask (unlike cropHoriz)
	imgs = [x.T for x in imgs]

def splitPage(imgname):
	img = misc.imread(imgname)
	img = rgb2gray(img)
	
	imgs = cropOne(img)
	imgs = [cropOne(x.T) for x in imgs]

	return imgs

def saveImgs(imgs, index=1, suffix=".png"):
	for x in imgs:
		if len(x) == 1:
			puzzle = x[0].T
			fname = "tsumego/tsumego" + str(index) + suffix
			misc.imsave(fname, puzzle)
			index += 1
		else:
			digits = [d.T for d in x[0:]]
			# You can do some OCR on each of the characters to read the filename
			continue
	return index

if __name__ == "__main__":
	pagesDir = "pages/"
	suffix = ".png"
	pages = [f for f in os.listdir(pagesDir) if suffix in f]
	pages = sorted(pages, key=lambda x: int(x.strip("page.png")))
	
	index = 1
	for p in pages:
		imgs = splitPage(os.path.join(pagesDir, p))
		index = saveImgs(imgs, index, suffix)
		print ("Next puzzle:", index)


