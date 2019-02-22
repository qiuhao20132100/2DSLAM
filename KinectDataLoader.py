import numpy as np
import os
import cv2
from PIL import Image
import math

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
        self.indexMap = np.array(np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0])))



        self.calibrationInv = np.linalg.inv(np.array([[585.05108211,0,242.94140713],
                                                      [0,585.05108211,315.83800193],
                                                      [0,0,1]]))
        # print(self.indexMap[0][0:10])
        pixelIndex = np.stack((self.indexMap[1], self.indexMap[0], np.ones(self.indexMap[0].shape)), axis = 2).reshape(-1,3)
        pixelIndex = np.swapaxes(pixelIndex,0,1)
        # print(pixelIndex.shape)
        self.pixelIndexMultipleCalibrationInv = self.calibrationInv.dot(pixelIndex)
        # print(self.pixelIndexMultipleCalibrationInv.shape)


        self.RocOneInv = np.array([[0, 0, 1, 0],
                                   [-1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 0, 1]])

        self.camera_position = np.array([0.18, 0.005, 0.36])
        camera_rpy = np.array([0.0, 0.36, 0.021])  # rad
        rotation_camera_cam2body_yaw = np.array([[np.cos(0.021), -np.sin(0.021), 0],
                                                      [np.sin(0.021), np.cos(0.021), 0],
                                                      [0, 0, 1]])

        rotation_camera_cam2body_pitch = np.array([[np.cos(0.36), 0, np.sin(0.36)],
                                                        [0, 1, 0],
                                                        [-np.sin(0.36), 0, np.cos(0.36)]])

        rotation_camera_cam2body_roll = np.array([[1, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 1]])

        self.rotation_camera_cam2body = np.dot(np.dot(rotation_camera_cam2body_yaw,
                                                      rotation_camera_cam2body_pitch),
                                               rotation_camera_cam2body_roll)
        self.transform_cam2body = np.hstack((self.rotation_camera_cam2body, self.camera_position.reshape(-1, 1)))
        self.transform_cam2body = np.vstack((self.transform_cam2body, np.array([[0, 0, 0, 1]])))
        # print(self.transform_cam2body)

        # direction = [-1,0,0]
        # angle = 0.36
        # q = [math.cos(angle / 2), math.sin(angle / 2) * direction[0],
        #      math.sin(angle / 2) * direction[1], math.sin(angle / 2) * direction[2]]
        # hat = np.array([
        #                 [0, -q[3], q[2]],
        #                 [q[3], 0, -q[1]],
        #                 [-q[2], q[1], 0]
        #                                 ])
        # qv = np.array(q[1:4]).reshape(3,1)
        # Eq = np.concatenate((qv, np.identity(3) * q[0] + hat), axis = 1)
        # Gq = np.concatenate((qv, np.identity(3) * q[0] - hat), axis = 1)
        # self.Rwc = Eq.dot(Gq.T)
        # self.qwc = np.array([0.005, 0.18, 0.36]).reshape(3,1)
        # self.R = np.concatenate((self.Rwc, self.qwc), axis = 1)
        # self.R = np.concatenate((self.R, np.array([0,0,0,1]).reshape(1,4)))

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

    def getDepth(self, rgb, disparity):
        # cv2.imshow('img',rgb)
        # cv2.waitKey()
        ddisparity = (-0.00304 * disparity + 3.31)
        depth = 1.03 / ddisparity
        rgbIndexI = np.round((self.indexMap[1] * 526.37 + ddisparity * (-4.5 * 1750.46) + 19276.0) / 585.051).astype(np.int32)
        rgbIndexJ = np.round((self.indexMap[0] * 526.37 + 16662.0) / 585.051).astype(np.int32)
        # index = np.stack((rgbIndexI, rgbIndexJ), axis = 2)
        depthAssignedToRGB = np.zeros(rgb.shape[0:2])

        for row in range(rgbIndexI.shape[0]):
            for col in range(rgbIndexI.shape[1]):
                if (rgbIndexI[row][col] >= 0 and rgbIndexI[row][col] < rgb.shape[0] and rgbIndexJ[row][col] >= 0 and rgbIndexJ[row][col] < rgb.shape[1]):
                    depthAssignedToRGB[rgbIndexI[row][col], rgbIndexJ[row][col]] = depth[row, col]

        depthAssignedToRGB[depthAssignedToRGB <= 0] = 0
        # cv2.rectangle(image, (380, 0), (400, 30), (0, 255, 0), 3)
        # print(depth[0:30,380:400])
        # checki = np.ceil((self.indexMap[0][0:30,380:400] * 526.37 + depth[0:30,380:400] * ((-4.5) * 1750.46) + 19276.0) / 585.051).astype(np.int32)
        # checkj = np.ceil((self.indexMap[1][0:30,380:400] * 526.37 + 16662.0) / 585.051).astype(np.int32)
        # print(checki)
        # cv2.imshow('image',image)
        # cv2.waitKey()
        return depthAssignedToRGB

    def convertOToBody(self, depthAssignedToRGB):
        Z = depthAssignedToRGB.reshape(1,-1)
        tmp = self.pixelIndexMultipleCalibrationInv * Z
        PostionInOptical = np.stack((tmp[0], tmp[1], tmp[2], np.ones(Z.shape[1])), axis=0)
        PostionInOptical = self.transform_cam2body.dot(self.RocOneInv.dot(PostionInOptical))
        # tmp = np.array(PostionInOptical[0,:])
        # PostionInOptical[0, :] = -PostionInOptical[1, :]
        # PostionInOptical[1, :] = tmp
        return PostionInOptical

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
    depthAssignedToRGB = data.getDepth(image, disparity_img)
    # cv2.imshow('img', image)
    # cv2.waitKey()
    # image[depthAssignedToRGB > 0] = [0,0,0]
    # cv2.imshow('img', image)
    # cv2.waitKey()
    print(data.convertOToBody(depthAssignedToRGB).shape)