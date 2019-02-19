import numpy as np

class IMUData(object):

    def __init__(self, address):
        with np.load(address) as data:
            self.time_stamps = data["time_stamps"]
            self.yawData = data["angular_velocity"][2]
        self.currIndex = 0

    def getOneYawData(self, index):
        return self.yawData[index]

    def reset(self):
        self.currIndex = 0

    def getOneYawDataByTime(self, time):
        while(self.currIndex < len(self.time_stamps) and self.time_stamps[self.currIndex] < time):
            self.currIndex += 1

        if (self.currIndex == len(self.time_stamps)):
            return self.yawData[len(self.yawData) - 1]
        elif self.currIndex == 0:
            return self.yawData[self.currIndex]
        else:
            if(abs(self.time_stamps[self.currIndex] - time) < abs(self.time_stamps[self.currIndex - 1] - time)):
                return self.yawData[self.currIndex]
            else:
                return self.yawData[self.currIndex - 1]