import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from typing import Any
import flip_api
import data as flip_tools

import re
import matplotlib.pyplot as plt
import multiprocessing as mp

import glob
import sys
import cv2
import numpy as np
from math import log10, sqrt
import imageio as io


def stripName(fname):
    return fname[fname.rfind('/') + 1 : fname.rfind('.')]

def stripExt(fname):
    return fname[0: fname.rfind('.')]

def BGR2Yxy(bgrMatrix):
    x = 0.4124 * bgrMatrix[:,:, 2] + 0.3576 * bgrMatrix[:,:, 1] + 0.1805 * bgrMatrix[:,:, 0]
    y = 0.2126 * bgrMatrix[:,:, 2] + 0.7152 * bgrMatrix[:,:, 1] + 0.0722 * bgrMatrix[:,:, 0]
    z = 0.0193 * bgrMatrix[:,:, 2] + 0.1192 * bgrMatrix[:,:, 1] + 0.9505 * bgrMatrix[:,:, 0]

    invSum = 1 / (x + y + z + 0.0001)

    result = np.zeros(bgrMatrix.shape, dtype=np.float32)
    result[:,:,0] = y
    result[:,:,1] = x * invSum
    result[:,:,2] = y * invSum
    return result

def Yxy2BGR(yxyMatrix):

    yy = (yxyMatrix[:,:, 0] / (yxyMatrix[:,:, 2] + 0.01))
    x = yxyMatrix[:,:, 1] * yy
    y = yxyMatrix[:,:, 0]
    z = (1 - yxyMatrix[:,:, 1] - yxyMatrix[:,:, 2]) * yy

    result = np.zeros(yxyMatrix.shape, dtype=np.float32)
    result[:,:,2] =  3.2410 * x - 1.5374 * y - 0.4986 * z
    result[:,:,1] = -0.9692 * x + 1.8760 * y + 0.0416 * z
    result[:,:,0] =  0.0556 * x - 0.2040 * y + 1.0567 * z
    return result

def Tonemap(inHDR, Lwhite = None, Key = None):
    Yxy = BGR2Yxy(inHDR)
    # Tonemap Reinhard using Lwhite
    imgLum = Yxy[:,:,0]

    if Key is None:
        Key = np.sum(np.log(imgLum + 0.0001))
        Key = np.exp(Key / (imgLum.shape[0] * imgLum.shape[1]))

    # Reinhard 2002 Eq.1-2
    imgLum = (0.18 / Key) * imgLum

    if Lwhite is None:
        Lwhite = np.max(imgLum)

    # Reinhard 2002 Eq.4
    lwSqr = Lwhite * Lwhite
    imgLumAdj = (imgLum * (1 + (imgLum / lwSqr))) / (1 + imgLum)

    # Only adjust luminance
    imgTM = Yxy
    imgTM[:,:,0] =  imgLumAdj
    imgTM = Yxy2BGR(imgTM)

    # Gamma
    imgTM = np.abs(imgTM) ** (1.0/2.2)

    imgTM = (imgTM * 255).clip(0, 255)
    imgTM = imgTM.astype(np.uint8)

    return (imgTM, Lwhite, Key)


class ToneMapFunctor:
    def __init__(self, region, Lwhite, Key):
        self.region = region
        self.LwhiteRef = Lwhite
        self.KeyRef = Key

    def __call__(self, samples) -> Any:
        image = cv2.imread(samples[1], cv2.IMREAD_UNCHANGED)

        roiImg = image[self.region[2]:self.region[3],
                       self.region[0]:self.region[1]]

        #sdr, lw, k = Tonemap(roiImg, Lwhite = self.LwhiteRef, Key = self.KeyRef)
        sdr, lw, k = Tonemap(roiImg, Key = self.KeyRef)
        #sdr, lw, k = Tonemap(roiImg)

        cv2.imwrite(stripExt(samples[1]) + "_tm.jpg", sdr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print("ImgDone: Key {} Lwhite {} '{}'".format(str(k), str(lw), samples[1]))

def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print("Not enough arguments!")
        return

    refFile : str = args[0]
    timeCount : float = float(args[1])

    intList = args[2].split(',')
    region = (int(intList[0]), int(intList[0]) + int(intList[2]),
              int(intList[1]), int(intList[1]) + int(intList[3]))

    checkFileRegexList : str = args[3:]

    fileDict = {}
    sortedDict={}
    i = 0
    for reg in checkFileRegexList:
        data = glob.glob(reg)
        print(reg)
        for d in data:
            seedSpp = re.findall(r"[+-]?\d+(?:\.\d+)?", d)
            seedSpp = float(seedSpp[-1])
            if seedSpp > timeCount:
                continue

            if i not in fileDict:
                fileDict[i] = {}

            fileDict[i][seedSpp] = d
        sortedDict[i] = dict(sorted(fileDict[i].items()))

        i = i + 1


    # Load Exr image and tone map it as well
    imRef = cv2.imread(refFile, cv2.IMREAD_UNCHANGED)
    roiRef = imRef[region[2]:region[3], region[0]:region[1]]
    roiRefSDR, Lwhite, Key = Tonemap(roiRef)

    cv2.imwrite(stripExt(refFile) + "_tm.jpg", roiRefSDR, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print("Reference: Key {} Lwhite {}".format(str(Key), str(Lwhite)))

    with mp.Pool(processes = mp.cpu_count()) as p:
        for d in sortedDict.items():
            print(d[0])
            p.map(ToneMapFunctor(region, Lwhite, Key), d[1].items())

if __name__ == "__main__":
    main()