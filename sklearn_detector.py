import cv2
import numpy as np
import os

from skimage.feature import hog
from sklearn.svm import SVC

    
cell_size = (8, 8)
block_size = (2, 2)
win_size = (4, 8)

nbins = 9

hogDesc = cv2.HOGDescriptor(
    (64, 128),
    (16, 16),
    cell_size,
    cell_size,
    nbins
)

def get_hog(img):
    fd = hog(img, orientations=nbins, pixels_per_cell=cell_size, cells_per_block=block_size)
    return fd


### Creating TrainingData

x_train = []
y_train = []

path = "training/cropped_human/"
dirs = os.listdir(path)

for item in dirs:
    fullpath = os.path.join(path, item)
    if os.path.isfile(fullpath):
        im = cv2.imread(fullpath)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        x_train.append(get_hog(gray))
        y_train.append(1)

path = "training/cropped_nonhuman/"
dirs = os.listdir(path)

for item in dirs:
    fullpath = os.path.join(path, item)
    if os.path.isfile(fullpath):
        im = cv2.imread(fullpath)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        x_train.append(get_hog(gray))
        y_train.append(0)


### Creating SVM
x_train = np.array(x_train)
y_train = np.array(y_train)

svm = SVC(kernel='linear')
svm.fit(x_train, y_train)

### Creating test data

# testing human

x_test = []
y_test = []

path = "testing/human"
dirs = os.listdir(path)

for item in dirs:
    fullpath = os.path.join(path, item)
    if os.path.isfile(fullpath):
        im = cv2.imread(fullpath)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        x_test.append(get_hog(gray))
        y_test.append(1)

path = "testing/nonhuman"
dirs = os.listdir(path)

for item in dirs:
    fullpath = os.path.join(path, item)
    if os.path.isfile(fullpath):
        im = cv2.imread(fullpath)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        x_test.append(get_hog(gray))
        y_test.append(0)

x_test = np.array(x_test)

y_pred = svm.predict(x_test)



from sklearn.metrics import det_curve, precision_recall_curve, PrecisionRecallDisplay, DetCurveDisplay, accuracy_score
import matplotlib.pyplot as plt

print(accuracy_score(y_test, y_pred))
PrecisionRecallDisplay.from_predictions(y_test, y_pred)
DetCurveDisplay.from_predictions(y_test,y_pred)
plt.show()
'''
labels = groundTruth.ravel()

fpr, fnr, thresholds = det_curve(labels, scores)
precision, recall, thresholds = precision_recall_curve(groundTruth, scores)
PrecisionRecallDisplay.from_predictions(labels, scores)
DetCurveDisplay.from_predictions(labels,scores)
print(precision, recall)

import matplotlib.pyplot as plt

plt.show()'''