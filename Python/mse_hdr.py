
from typing import Any
import flip_api
import data as flip_tools
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import re
import matplotlib.pyplot as plt
import glob
import sys

import cv2
import numpy as np

from math import log10, sqrt

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Not enough arguments!")
        return

    refFile : str = args[0]
    compfiles = args[1:]

    imRef = flip_tools.read_exr(refFile)
    #imRef = cv2.imread(refFile)
    #imRef = cv2.normalize(imRef, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    roiRef = imRef

    for cPath in compfiles:
        image = flip_tools.read_exr(cPath)
        #image = cv2.imread(cPath)
        #image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        roiImg = image

        mse = (roiImg - roiRef) ** 2
        mseOut = np.mean(mse)

        if roiRef.shape != imRef.shape:
            print("The images are not the same size")
            return
        else:
            print(mseOut)

if __name__ == "__main__":
    main()