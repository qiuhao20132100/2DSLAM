import numpy as np

class OdometryData(object):
    def __init__(self, address):
        with np.load(address) as data:
            self.encoder_counts = data["counts"].T  # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"]  # encoder time stamps
            self.length = len(self.encoder_stamps)

        # self.currIndex = 0
        # self.count = [0,0,0,0]
        # if self.encoder_stamps is not None and len(self.encoder_stamps) > 0:
        #     self.curr = self.encoder_stamps[0]



    # def getClosedCount(self, t):
    #     while (self.currIndex < len(self.encoder_stamps) and self.encoder_stamps[self.currIndex] < t):
    #         self.count[0] += self.encoder_counts[self.currIndex][0]
    #         self.count[1] += self.encoder_counts[self.currIndex][1]
    #         self.count[2] += self.encoder_counts[self.currIndex][2]
    #         self.count[3] += self.encoder_counts[self.currIndex][3]
    #         self.currIndex += 1
    #     return self.count



