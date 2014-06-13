ColorCheckerDetector
====================

This python module implements a function 'findStandard' that takes an OpenCV RGB image and outputs a projection of a Macbeth 6x4 color checker. 

findStandard takes one argument (the image) and outputs a projection of the standard. 
It will run fairly slowly for large images at 6MP or higher. If the downsample flag is set to true it will downsample to X pixel width of 2000
It WILL fail when there are more than one standard in the picture.
It is fairly robust to even partially  covered or irregularly shaped standards to a small extent. 
Running this module by itself will find the standard for all jpg files in the same folder and save them to disk in the same folder

Usage
==================
import cv2
from standardDetector import *
standard = findStandard(cv2.imread('file.jpg'))
