import cv2
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, accuracy_score, DetCurveDisplay
import matplotlib.pyplot as plt
import time

### Create HOG descriptor


#training_time_taken = 0
#training_start_time = time.time()

## default HOG descriptor

cell_size = (8, 8)
block_size = (16, 16)
image_size = (64, 128)

nbins = 9

hogDesc = cv2.HOGDescriptor(
    image_size,
    block_size,
    cell_size,
    cell_size,
    nbins
)

def get_hog(img):
    hog_feats = hogDesc.compute(img)
    return hog_feats

### Creating TrainingData

def create_svm():
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


    y_train = np.asarray(y_train)
    x_train = np.asmatrix(x_train)


    ### Creating SVM

    svm = cv2.ml.SVM.create()

    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

    #training_time_taken = time.time() - training_start_time ## for finding time to generate svm
    #print(f"Time taken to create SVM: {training_time_taken}")

    #svm.save("base_detector.xml") ## create xml for gui program

    return svm

def test_svm(svm : cv2.ml.SVM):
    x_test = []
    y_true = []

    testing_start_time = time.time()

    path = "testing/human"
    dirs = os.listdir(path)

    for item in dirs:
        fullpath = os.path.join(path, item)
        if os.path.isfile(fullpath):
            im = cv2.imread(fullpath)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x_test.append(get_hog(gray))
            y_true.append(1)

    path = "testing/nonhuman"
    dirs = os.listdir(path)

    for item in dirs:
        fullpath = os.path.join(path, item)
        if os.path.isfile(fullpath):
            im = cv2.imread(fullpath)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x_test.append(get_hog(gray))
            y_true.append(0)

    x_test = np.asmatrix(x_test)

    _, y_pred = svm.predict(x_test, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)

    #testing_time_taken = time.time() - testing_start_time ## for finding time to test images
    #print(f"Time taken to test data: {testing_time_taken}") ## 
    #print(f"Total time: {testing_start_time + training_time_taken}") ##

    ## flipped so that output matches labels
    y_pred = -y_pred.ravel()
    y_binary = (y_pred > 0).astype(int)

    y_true = np.asarray(y_true)

    #print(y_true, y_pred) ## for testing

    accuracy = accuracy_score(y_true, y_binary)
    print(f"Accuracy: {accuracy}")
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_pred)

    #plt.show()
    return accuracy, disp

def test_images(svm, folder):
    x_test = []

    path = folder
    dirs = os.listdir(path)

    filenames = []

    for item in dirs:
        fullpath = os.path.join(path, item)
        if os.path.isfile(fullpath):
            im = cv2.imread(fullpath)
            im = cv2.resize(im, (64, 128), interpolation = cv2.INTER_LINEAR)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x_test.append(get_hog(gray))
            filenames.append(item)
            
    x_test = np.asmatrix(x_test)
    _, y_pred = svm.predict(x_test)
    return y_pred, filenames

if __name__ == '__main__':
    
    ## for ablation study

    bin_nums = [2,3,9,24]

    for bins in bin_nums:
        nbins = bins

        hogDesc = cv2.HOGDescriptor(
            image_size,
            block_size,
            cell_size,
            cell_size,
            nbins
        )

        svm = create_svm()
        accuracy_0, disp_0 = test_svm(svm)
        plt.title(str(bins) + " bins")

    plt.show()