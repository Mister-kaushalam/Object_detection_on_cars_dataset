#importing necessary packages
from __future__ import print_function
from object_detection.utils import dataset
from object_detection.utils.conf import Conf
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import argparse
import pickle

#construct argumnet parser and parse the command line arguments
ap =argparse.ArgumentParser()
ap.add_argument("-c","--conf", required= True, help="Path to configuration file")
ap.add_argument("-n", "--hard-negatives", type=int, default=-1,
	help="flag indicating whether or not hard negatives should be used")
args = vars(ap.parse_args())

#load the configuration file and initialize the dataset

print("[INFO] loading dataset...")
conf = Conf(args["conf"])
(data, labels) = dataset.load_dataset(conf["features_path"], "features")

# check to see if the hard negatives flag was supplied
if args["hard_negatives"] > 0:
	print("[INFO] loading hard negatives...")
	(hardData, hardLabels) = dataset.load_dataset(conf["features_path"], "hard_negatives")
	data = np.vstack([data, hardData])
	labels = np.hstack([labels, hardLabels])

#train the classifier
print("[INFO] training classifier..")
model = SVC(kernel="linear", C=conf["C"], gamma=conf["gamma"], probability=True, random_state=42)
model.fit(data,labels)

#dump the classifier to file
print("[INFO] dumping the classifier...")
f=open(conf["classifer_path"], "wb")
f.write(pickle.dumps(model))
f.close()


#testing the parameters with grid_search
#grid = GridSearchCV(SVC(kernel="linear"),param_grid=conf["grid_param"],refit=True,verbose=1, n_jobs=-1)
#grid.fit(data,labels)


#best parameter estimation
'''
print("[GRID RESULT] Best parameters: ", grid.best_params_)
print("[GRID RESULT] best estimator: ", grid.best_estimator_)

already complied results:

[GRID RESULT] Best parameters:  {'C': 0.1, 'gamma': 1}
[GRID RESULT] best estimator:  SVC(C=0.1, gamma=1, kernel='linear')
'''

