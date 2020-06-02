'''
we'll be using sliding windows, an object detection tool that is used to “slide” over an image from left-to-right and top-to-bottom. 
At each window position, HOG features are extracted and then passed on to our classifier to determine 
if an object of interest resides within that particular window. However, we need to know the apporopriate size of our sliding window.
The size of our sliding window affects the HOG descriptor. Instead of making guess we'll try to figure out by exploring the dimentions of our object
annotations in order to make a more informed decision.
'''

#import necessary packages
from __future__ import print_function
from scipy import io
import numpy as np
import argparse
import glob
from object_detection.utils.conf import Conf

#contructing an argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c","--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

#load the configuration file and intialize the list of widths and heights
conf=Conf(args['conf'])
print(conf["image_annotations"])
widths =[]
heights = []

#loop over all the annotations paths
for p in glob.glob(conf["image_annotations"] + "/*.mat"):
    #load all the bounding box associated with the path and update the width and height
    (y,h,x,w) = io.loadmat(p)["box_coord"][0]
    widths.append(w-x)
    heights.append(h-y)

#compute the average of widths and heights lists
(avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
print("[INFO] avg. width: {:.2f}".format(avgWidth))
print("[INFO] avg. height: {:.2f}".format(avgHeight))
print("[INFO] aspect ratio: {:.2f}".format(avgWidth / avgHeight))
