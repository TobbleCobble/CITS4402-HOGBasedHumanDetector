import tkinter as tk
import tkinter.filedialog as filedialog
from pandas import DataFrame
from PIL import ImageOps, ImageTk, Image
import cv2

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

    return path, prediction


def load_images():
    global folder
    global filenames
    global predictions
    index = 0

    folder = filedialog.askdirectory()

    filenames, predictions = test_images(folder)
    
    df = DataFrame({'filename': filenames, 'prediction': predictions})
    df.to_excel('predictions.xlsx', index=False)

    show_image(0)

    images.pack()
    get_started.pack_forget()

def next_image():
    global index
    global filenames

    index += 1
    index = index%len(filenames)
    show_image(index)

def prev_image():
    global index
    global filenames
    
    index -= 1
    index = index%len(filenames)
    show_image(index)
    

def show_image(index):
    global folder
    global filenames
    global predictions

    im = Image.open(folder+'/'+filenames[index])
    im = ImageTk.PhotoImage(im)
    display_image.config(image=im)
    display_image.img = im

    label = "human"
    if predictions[index] != 1:
        label = "nonhuman"

    prediction_text.config(text=label)

images = tk.Frame(root)

## display image
display_image = tk.Label(images)
display_image.grid(row=0, column=1)

## buttons to swap between images
next_button = tk.Button(images, text=">", command=next_image)
next_button.grid(row=0, column=2, padx=10)

prev_button = tk.Button(images, text="<", command=prev_image)
prev_button.grid(row=0, column=0, padx=10)

## text to say human or not
prediction_text = tk.Label(images, text="human")
prediction_text.grid(row=1, column=1)

## Loading folder
get_started = tk.Label(root, text="load images to get started")
get_started.pack(pady=10, padx=15)

button = tk.Button(root, text="load images", command=load_images)
button.pack(pady=10)

images.pack_forget()

root.title("HoG Human Detector")
root.mainloop()