import numpy as np
import sys
import os
from enum import Enum

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import flip
import cv2 as cv
import imageio as io
import math

class NoiseMode(Enum):
    COLOR = 0
    LUMINANCE = 1

def main():
    args = sys.argv[1:]
    if(len(args) != 4):
        print("Not enough arguments!")
        return

    file : str = args[0]
    stdDev : float = float(args[1])
    outName : str = args[2]
    noiseMode: NoiseMode = NoiseMode[args[3]]

    img = cv.imread(file, cv.IMREAD_UNCHANGED)

    if(noiseMode == NoiseMode.LUMINANCE):
        img = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

        samples = np.random.normal(0, stdDev, size=img.shape[0:2])
        img[:,:,0] = img[:,:,0] + samples
        out = cv.cvtColor(img, cv.COLOR_YCR_CB2RGB)
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        samples = np.random.normal(0, stdDev, size=img.shape)
        out = img + samples


    out = np.float32(out)

    io.imwrite(outName + str(stdDev) + '.exr', out)

if __name__ == "__main__":
    main()