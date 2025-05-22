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
index = 0
folder = ""

#df = DataFrame({'filename': filenames, 'prediction': predictions})

def test_images(folder):
    prediction, path = detector.test_images(svm, folder)
    prediction = list(prediction.ravel())
    filenames.append(path)
    predictions.append(prediction)


def load_images():
    global folder
    index = 0
    folder = filedialog.askdirectory()
    test_images(folder)
    
    df = DataFrame({'filename': filenames[0], 'prediction': predictions[0]})
    df.to_excel('test.xlsx', index=False)

    image_in = tk.PhotoImage(file=folder+'/'+filenames[0][0])
    display_image.config(image=image_in)
    display_image.img = image_in

    label = "human"
    if predictions[0][0] != 1:
        label = "nonhuman"

    prediction_text.config(text=label)
    images.pack()

def next_image():
    global folder
    global index
    index += 1
    index = index%len(filenames[0])
    image_in = tk.PhotoImage(file=folder+'/'+filenames[0][index])
    display_image.config(image=image_in)
    display_image.img = image_in

    label = "human"
    if predictions[0][index] != 1:
        label = "nonhuman"

    prediction_text.config(text=label)

def prev_image():
    global folder
    global index
    index -= 1
    index = index%len(filenames[0])
    image_in = tk.PhotoImage(file=folder+'/'+filenames[0][index])
    display_image.config(image=image_in)
    display_image.img = image_in

    label = "human"
    if predictions[0][index] != 1:
        label = "nonhuman"

    prediction_text.config(text=label)


images = tk.Frame(root)

display_image = tk.Label(images)
display_image.grid(row=0, column=1)

## need text to say human or not
prediction_text = tk.Label(images, text="human")
prediction_text.grid(row=1, column=1)
## need buttons to swap between images
next_button = tk.Button(images, text=">", command=next_image)
next_button.grid(row=0, column=2)
prev_button = tk.Button(images, text="<", command=prev_image)
prev_button.grid(row=0, column=0)

button = tk.Button(root, text="load images", command=load_images)
button.pack()
images.pack_forget()

root.mainloop()