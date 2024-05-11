from typing import Any
import flip_api
import data as flip_tools

import os
import re
import matplotlib.pyplot as plt
import multiprocessing as mp

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import glob
import sys
import cv2
import numpy as np
from math import log10, sqrt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def stripName(fname):
    return fname[fname.rfind('/') + 1: fname.rfind('.')]

def print_initialization_information(pixels_per_degree, hdr, tone_mapper=None, start_exposure=None, stop_exposure=None, num_exposures=None):
	"""
	Prints information about the metric invoked by FLIP

	:param pixels_per_degree: float indicating number of pixels per degree of visual angle
	:param hdr: bool indicating that HDR images are evaluated
	:param tone_mapper: string describing which tone mapper HDR-FLIP assumes
	:param start_exposure: (optional) float indicating the shortest exposure HDR-FLIP uses
	:param stop_exposure: (optional) float indicating the longest exposure HDR-FLIP uses
	:param number_exposures: (optional) integer indicating the number of exposure HDR-FLIP uses
	"""
	print("Invoking " + ("HDR" if hdr else "LDR") + "-FLIP")
	print("\tPixels per degree: %d" % round(pixels_per_degree))
	if hdr == True:
		tone_mapper = tone_mapper.lower()
		if tone_mapper == "hable":
			tm = "Hable"
		elif tone_mapper == "reinhard":
			tm = "Reinhard"
		else:
			tm = "ACES"
		print("\tAssumed tone mapper: %s" % tm)
		print("\tStart exposure: %.4f" % start_exposure)
		print("\tStop exposure: %.4f" % stop_exposure)
		print("\tNumber of exposures: %d" % num_exposures)
	print("")

def set_start_stop_num_exposures(reference, start_exp=None, stop_exp=None, num_exposures=None, tone_mapper="aces"):
	# Set start and stop exposures
	if start_exp == None or stop_exp == None:
		start_exposure, stop_exposure = flip_api.compute_exposure_params(reference, tone_mapper=tone_mapper)
		if start_exp is not None: start_exposure = start_exp
		if stop_exp is not None: stop_exposure = stop_exp
	else:
		start_exposure = start_exp
		stop_exposure = stop_exp
	assert start_exposure <= stop_exposure

	# Set number of exposures
	if start_exposure == stop_exposure:
		num_exposures = 1
	elif num_exposures is None:
		num_exposures = int(max(2, np.ceil(stop_exposure - start_exposure)))
	else:
		num_exposures = num_exposures

	return start_exposure, stop_exposure, num_exposures

class MSEFunctor:
    def __init__(self, region, roiRef, refFile):
        self.region = region
        self.roiRef = roiRef
        self.refFile = refFile

    def __call__(self, samples) -> Any:
        image = flip_tools.read_exr(samples[1])
        image = image[..., ::-1].copy()

        roiImg = image[self.region[2]:self.region[3],
                       self.region[0]:self.region[1]]

        mse = (roiImg - self.roiRef) ** 2

        mseOut = np.mean(mse)
        print((samples[0],mseOut))
        return (samples[0],mseOut)

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

    imRef = flip_tools.read_exr(refFile)
    imRef = imRef[..., ::-1].copy()

    roiRef = imRef[region[2]:region[3],
                   region[0]:region[1]]

    varianceListAll = []

    #print(fileDict)
    #print(sortedDict)

    with mp.Pool(processes = mp.cpu_count()) as p:
        for d in sortedDict.items():
            print(d[0])
            results = p.map(MSEFunctor(region, roiRef, refFile), d[1].items())
            sorted(results)
            varianceListAll.append(results)

    # print(varianceListAll[0])
    # print(varianceListAll[1])
    # print(varianceListAll[2])
    print(len(varianceListAll))
    i = 0
    for v in varianceListAll:
        np.savetxt("mse_convergence" + str(i) + ".csv", v, delimiter=",", fmt='%1.7f')
        i+=1


    # iotaArr = list(mydict.keys());len(sortedDict[i])
    # iota = np.arange(1, len(sortedDict[i]) + 1, 1)

    colors = np.asarray([[230, 25, 75], [60, 180, 75], [255, 225, 25],
                         [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
                         [240, 50, 230]])
    colors = colors / 255.

    i = 0
    for v in varianceListAll:
        plt.plot(*zip(*v), color=colors[i], label=str(i))
        i+=1

    plt.legend()
    #plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    main()