import numpy as np
import os
import cv2
from PIL import Image
class KinectData(object):
    def __init__(self, prefix, dataSet):
        with np.load(prefix + ("Kinect%d.npz" % dataSet)) as data:
            self.DisparityStamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
            self.RGBStamps = data["rgb_time_stamps"]  # acquisition times of the rgb images
        self.RGBAddress = prefix + ("RGB%d" % dataSet) + os.path.sep
        self.DisparityAddress = prefix + ("Disparity%d" % dataSet) + os.path.sep
        self.currRGBIndex = 0
        self.currDisparityIndex = 0
        self.dataSet = dataSet
        shape = self.getOneDisparityDataByTime(0).shape
        self.indexMap = np.zeros((2, shape[0], shape[1]))
        for row in range(shape[0]):
            for col in range(shape[1]):
                self.indexMap[0,row,col] = row
                self.indexMap[1,row,col] = col


    def reset(self):
        self.currRGBIndex = 0
        self.currDisparityIndex = 0

    def getOneRGBDataByTime(self, time):
        while(self.currRGBIndex < len(self.RGBStamps) and self.RGBStamps[self.currRGBIndex] < time):
            self.currRGBIndex += 1

        if (self.currRGBIndex == len(self.RGBStamps)):
            return self.readRGBFile(self.currRGBIndex - 1)
        elif self.currRGBIndex == 0:
            return self.readRGBFile(self.currRGBIndex)
        else:
            if(abs(self.RGBStamps[self.currRGBIndex] - time) < abs(self.RGBStamps[self.currRGBIndex - 1] - time)):
                return self.readRGBFile(self.currRGBIndex)
            else:
                return self.readRGBFile(self.currRGBIndex - 1)

    def readRGBFile(self, index):
        fileName = "rgb%d_%d.png" % (self.dataSet, index + 1)
        return cv2.imread(self.RGBAddress + fileName)

    def getOneDisparityDataByTime(self, time):
        while (self.currDisparityIndex < len(self.DisparityStamps) and self.DisparityStamps[self.currDisparityIndex] < time):
            self.currDisparityIndex += 1

        if (self.currDisparityIndex == len(self.DisparityStamps)):
            return self.readDisparityFile(self.currDisparityIndex - 1)
        elif self.currDisparityIndex == 0:
            return self.readDisparityFile(self.currDisparityIndex)
        else:
            if (abs(self.DisparityStamps[self.currDisparityIndex] - time) < abs(self.DisparityStamps[self.currDisparityIndex - 1] - time)):
                return self.readDisparityFile(self.currDisparityIndex)
            else:
                return self.readDisparityFile(self.currDisparityIndex - 1)

    def readDisparityFile(self, index):
        fileName = "disparity%d_%d.png" % (self.dataSet, index + 1)
        img = Image.open(self.DisparityAddress + fileName)
        return np.array(img.getdata(), np.uint16).reshape(img.size[1], img.size[0])


    def getDepthAndRGB(self, rgb, disparity):
        # cv2.imshow('img',rgb)
        # cv2.waitKey()
        ddisparity = (-0.00304 * disparity + 3.31)
        depth = 1.03 / ddisparity
        rgbIndexI = np.ceil((self.indexMap[0] * 526.37 + depth * ((-4.5) * 1750.46) + 19276.0) / 585.051).astype(np.int32)
        rgbIndexJ = np.ceil((self.indexMap[1] * 526.37 + 16662.0) / 585.051).astype(np.int32)
        index = np.stack((rgbIndexI, rgbIndexJ), axis = 2)
        image = np.zeros(rgb.shape, dtype = np.uint8)
        for row in range(index.shape[0]):
            for col in range(index.shape[1]):
                image[row,col] = rgb[rgbIndexI[row][col],rgbIndexJ[row][col]]
        image[np.logical_or(
            np.logical_or(
                np.logical_or(rgbIndexJ < 0, rgbIndexJ >= index.shape[1]),
                np.logical_or(rgbIndexI < 0, rgbIndexI >= index.shape[0])),
            depth <= 0)
        ] = [0, 0, 0]
        # cv2.rectangle(image, (380, 0), (400, 30), (0, 255, 0), 3)
        # print(depth[0:30,380:400])
        # checki = np.ceil((self.indexMap[0][0:30,380:400] * 526.37 + depth[0:30,380:400] * ((-4.5) * 1750.46) + 19276.0) / 585.051).astype(np.int32)
        # checkj = np.ceil((self.indexMap[1][0:30,380:400] * 526.37 + 16662.0) / 585.051).astype(np.int32)
        # print(checki)
        # cv2.imshow('image',image)
        # cv2.waitKey()
        return image
if __name__ == '__main__':
    data = KinectData('data/dataRGBD/', 20)
    # print(data.DisparityAddress)
    # print(data.RGBAddress)
    # print(data.currRGBIndex)
    # print(data.currDisparityIndex)
    # print(data.dataSet)
    # print(data.DisparityStamps)
    # print(data.RGBStamps)
    # print(data.DisparityStamps[0])
    # print(data.DisparityStamps[1])
    # print(data.RGBStamps[0])
    image = data.getOneRGBDataByTime(0)
    disparity_img = data.getOneDisparityDataByTime(0)
    mixImag = data.getDepthAndRGB(image, disparity_img)
    cv2.imshow('img',mixImag)
    cv2.waitKey()