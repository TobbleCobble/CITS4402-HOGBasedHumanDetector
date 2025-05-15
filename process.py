from PIL import Image
import os

### used to convert images to 64x128

path = "0/"
dirs = os.listdir(path)

def crop():
    for item in dirs:
        fullpath = os.path.join(path, item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((im.width // 2 - 32, im.height // 2 - 64, im.width // 2 - 32 + 64, im.height // 2 - 64 + 128))
            imCrop.save(f + '.png', "PNG", quality=100)

path = "cropped_nonhuman/"
dirs = os.listdir(path)

def resize():
    for item in dirs:
        fullpath = os.path.join(path, item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.resize((64, 128))
            imCrop.save(f + '.png', "PNG", quality=100)


resize()