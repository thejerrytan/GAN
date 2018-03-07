import os, glob, tables
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from gan import IMG_WIDTH, IMG_HEIGHT

ROOT = os.getcwd()

def generate_dataset():    
    FOLDER = "car_images/"
    FILENAME = "car_dataset_%dx%d.h5" % (IMG_WIDTH, IMG_HEIGHT)
    TARGET_SIZE = 50000
    os.chdir(ROOT)
    fd = tables.open_file(FILENAME, mode='w')
    atom = tables.Float64Atom()
    filters = tables.Filters(complevel=5, complib='blosc')
    dataset = fd.create_earray(fd.root, 'data', atom, (0, 3, IMG_WIDTH, IMG_HEIGHT), filters=filters, expectedrows=TARGET_SIZE)
    
    os.chdir(ROOT + "/" + FOLDER)
    count = 0
    for f in glob.glob("*.jpg"):
        img = Image.open(f)
        count += 1
        print("%d : %s" % (count, f))
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        arr = np.asarray(img)
        arr = np.reshape(arr, (1, arr.shape[2], arr.shape[1], arr.shape[0]))
        dataset.append(arr)

    fd.close()

if __name__ == "__main__":
    generate_dataset()
