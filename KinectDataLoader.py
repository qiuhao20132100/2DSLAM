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

        #Parameters related to Kinect and Robot
        shape = self.getOneDisparityDataByTime(0).shape
        self.indexMap = np.array(np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0])))#(self.indexMap[0])#col index (self.indexMap[1])#row index
        self.calibrationInv = np.linalg.inv(np.array([[585.05108211,0,242.94140713],
                                                      [0,585.05108211,315.83800193],
                                                      [0,0,1]]))
        self.RocOneInv = np.array([[0, 0, 1, 0],
                                   [-1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 0, 1]])

        self.camera_position = np.array([0.18, 0.005, 0.36])
        cameraRPY = np.array([0.0, 0.36, 0.021])  # rad
        yawRotation = np.array([[np.cos(cameraRPY[2]), -np.sin(cameraRPY[2]), 0],
                                [np.sin(cameraRPY[2]), np.cos(cameraRPY[2]), 0],
                                [0, 0, 1]])

        pitchRotation = np.array([[np.cos(cameraRPY[1]), 0, np.sin(cameraRPY[1])],
                                  [0, 1, 0],
                                  [-np.sin(cameraRPY[1]), 0, np.cos(cameraRPY[1])]])

        rollRotation = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

        self.rotation_camera_cam2body = np.dot(np.dot(yawRotation, pitchRotation),rollRotation)
        self.transfromCamToBody = np.hstack((self.rotation_camera_cam2body, self.camera_position.reshape(-1, 1)))
        self.transfromCamToBody = np.vstack((self.transfromCamToBody, np.array([[0, 0, 0, 1]])))

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

    def getDepthAndIndex(self, rgb, disparity):
        i = self.indexMap[0]
        j = self.indexMap[1]
        ddisparity = (-0.00304 * disparity + 3.31)
        depth = (1.03 / ddisparity).reshape(-1, 1)
        rgbi = np.round((i * 526.37 + ddisparity * (-4.5 * 1750.46) + 19276.0) / 585.051).astype(np.int32).reshape(-1, 1)
        rgbj = np.round((j * 526.37 + 16662.0) / 585.051).astype(np.int32).reshape(-1, 1)
        mask = np.logical_and ( depth >= 0,
                                np.logical_and(
                                  np.logical_and(rgbi >= 0, rgbi <= rgb.shape[1]),
                                  np.logical_and(rgbj >= 0, rgbj <= rgb.shape[0])
                                )
                              )

        rgbi = rgbi[mask]
        rgbj = rgbj[mask]
        depth = depth[mask]
        return depth, rgbi, rgbj

    def convertPixelToBody(self, pixels, depth):
        positionOptical = depth.reshape(1, -1) * np.dot(self.calibrationInv, pixels)
        positionOptical = np.vstack((positionOptical, np.ones_like(depth.reshape(1, -1))))
        return np.dot(self.RocOneInv, positionOptical)

if __name__ == '__main__':
    data = KinectData('data/dataRGBD/', 20)
    # # print(data.DisparityAddress)
    # # print(data.RGBAddress)
    # # print(data.currRGBIndex)
    # # print(data.currDisparityIndex)
    # # print(data.dataSet)
    # # print(data.DisparityStamps)
    # # print(data.RGBStamps)
    # # print(data.DisparityStamps[0])
    # # print(data.DisparityStamps[1])
    # # print(data.RGBStamps[0])
    # file_name = "./data/dataRGBD/Disparity{}/disparity{}_{}.png".format(20, 20, 1)
    #
    # img_test = cv2.imread(file_name, -1)
    # type(img_test)
    # print(img_test.shape)
    # image = data.getOneRGBDataByTime(0)
    # disparity_img = data.getOneDisparityDataByTime(0)
    # depth, rgbI, rgbJ = data.getDepthAndIndex(image, disparity_img)
    # print(depth.shape)
    # print(rgbI.shape)
    # print(rgbJ.shape)
    # # cv2.imshow('img', image)
    # # cv2.waitKey()
    # # image[depthAssignedToRGB > 0] = [0,0,0]
    # # cv2.imshow('img', image)
    # # cv2.waitKey()
    # # print(data.convertOToBody(depthAssignedToRGB).shape)