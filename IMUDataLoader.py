import numpy as np
from lowPassFilter import butter_lowpass_filter

class IMUData(object):

    def __init__(self, address):
        with np.load(address) as data:
            self.time_stamps = data["time_stamps"]

            # filter the data
            self.yawData = data["angular_velocity"][2]
            order = 1
            fs = 1000  # sample rate, Hz
            cutoff = 10  # desired cutoff frequency of the filter, Hz
            self.yawData = butter_lowpass_filter(self.yawData, cutoff, fs, order)

        self.currIndex = 0

    def getOneYawData(self, index):
        return self.yawData[index]

    def reset(self):
        self.currIndex = 0

    def getOneYawDataByTime(self, time):
        while(self.currIndex < len(self.time_stamps) and self.time_stamps[self.currIndex] < time):
            self.currIndex += 1

        if (self.currIndex == len(self.time_stamps) or self.currIndex == len(self.time_stamps) - 1):
            return (self.yawData[len(self.yawData) - 1] + self.yawData[len(self.yawData) - 2] / 2.0)
        elif self.currIndex == 0:
            return (self.yawData[self.currIndex] + self.yawData[self.currIndex + 1] / 2.0)
        else:
            if(abs(self.time_stamps[self.currIndex] - time) < abs(self.time_stamps[self.currIndex - 1] - time)):
                return (self.yawData[self.currIndex - 1] + self.yawData[self.currIndex] + self.yawData[self.currIndex + 1])/3.0
            else:
                return (self.yawData[self.currIndex - 2] + self.yawData[self.currIndex - 1] + self.yawData[self.currIndex])/3.0

