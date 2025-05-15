import tkinter as tk
import cv2

import detector

root = tk.Tk()

svm = cv2.ml.SVM.load("base_detector.xml")
print(detector.test_image(svm, "testing/human/413_0.png"))