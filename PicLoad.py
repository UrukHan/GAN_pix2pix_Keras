# Importing Libraries
import cv2
import os
from cv2 import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

# Class for working with the image database
class PICLOAD():
    # Class initialization
    def __init__(self):
        pass

    # Function of returning images through the generator
    def l_files(self, basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains = None):
        # Walk through the structure of the catalogs
        for (rootDir, dirNames, filenames) in os.walk(basePath):
            # Loop through the filenames of the current directory
            for filename in filenames:
                if contains is not None and filename.find(contains) == -1:
                    continue
                # Determine the extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # Image check and processing
                if ext.endswith(validExts):
                    # Returning an image through the generator
                    imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                    yield imagePath

    # Returning an image # Returning a set of directory and subdirectory files via the generator  
    def l_images(self, basePath, contains=None):
        return self.l_files(basePath, validExts = (".jpg", ".jpeg", ".png", ".bmp"), contains = contains)

    # Image upload function        
    def load_img(self, basePath, img_size):
        images = []
        basePath = list(self.l_images(basePath))
        for path in basePath:
            if not('OSX' in path):
                path = path.replace('\\','/')
                image = cv2.imread(path) 
                image = cv2.resize(image, img_size)
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return images 
    
    # The function of displaying pictures in a row
    def show_imeges(self, img):
        _,ax = plt.subplots(1, 3, figsize = (10, 30)) 
        for i in range(3):
            ax[i].imshow(img[i])
        return

    