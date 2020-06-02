#importing the necessary packages
from object_detection.object_detection.nms import non_max_suppression
from object_detection.object_detection.objectdetector import ObjectDetector
from object_detection.descriptors.HOG import HOG
from object_detection.utils.conf import Conf

import numpy as np
import imutils
import argparse
import cv2
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to be classified")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

#load the classifier, then intialize the HOG descriptor and the object detector
model = pickle.loads(open(conf["classifer_path"], "rb").read())
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
	cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"], block_norm="L1")
od = ObjectDetector(model, hog)

#load the image and convert it into grayscale
image =cv2.imread(args["image"])
image = imutils.resize(image, width=min(260, image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#detect objects in the image and apply non-maxima suppression to the bounding boxes
(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
	pyramidScale=conf["pyramid_scale"], minProb=conf["min_probability"])

pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
orig = image.copy()

#loop over the original bounding boxes and draw them

for (startX, startY, endX, endY) in pick:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

#show the output image
cv2.imshow("original", orig)
cv2.imshow("Image", image)
cv2.waitKey(0)