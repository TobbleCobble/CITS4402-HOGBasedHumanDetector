import tkinter as tk
import tkinter.filedialog as filedialog
from pandas import DataFrame
import numpy as np
from PIL import Image, ImageTk
import cv2
import os

import detector

root = tk.Tk()

svm = cv2.ml.SVM.load("base_detector.xml")
#print(detector.test_images(svm, "testing/human/413_0.png"))

filenames = []
predictions = []

#df = DataFrame({'filename': filenames, 'prediction': predictions})

def test_images(folder):
    prediction, path = detector.test_images(svm, folder)
    prediction = list(prediction.ravel())
    filenames.append(path)
    predictions.append(prediction)


def load_images():
    folder = filedialog.askdirectory()
    test_images(folder)
    
    df = DataFrame({'filename': filenames[0], 'prediction': predictions[0]})
    df.to_excel('test.xlsx', index=False)

    image_in = tk.PhotoImage(file=folder+'/'+filenames[0][0])
    display_image.config(image=image_in)
    display_image.img = image_in


display_image = tk.Label(root, image=tk.PhotoImage(file='person_and_bike_111.png'))
display_image.img = tk.PhotoImage(file='person_and_bike_111.png')
display_image.pack()

## need text to say human or not
## need buttons to swap between images

button = tk.Button(root, text="load images", command=load_images)
button.pack()

root.mainloop()