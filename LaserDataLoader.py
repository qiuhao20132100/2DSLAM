import numpy as np

class LaserData(object):

    def __init__(self, address):
        self.lidarPosToRobot = np.array([0, 514.35 / 10000, (298.33 - 330.20 / 2) / 10000])
        with np.load(address) as data:
            self.lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
            self.lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
            self.lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
            self.lidar_range_min = data["range_min"]  # minimum range value [m]
            self.lidar_range_max = data["range_max"]  # maximum range value [m]
            self.lidar_ranges = data["ranges"].T  # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
            degrees = np.arange(self.lidar_angle_min[()], self.lidar_angle_max[()] + self.lidar_angle_increment[0][0] * 0.1, self.lidar_angle_increment[0][0], dtype = np.float64)
            allX = -np.sin(degrees) * self.lidar_ranges #index of measure * index of beam
            allY = np.cos(degrees) * self.lidar_ranges
            allZ = np.zeros(allX.shape)
            self.data = np.swapaxes(np.array([allX, allY, allZ]), 0, 1)
            self.mask = (np.array([self.lidar_ranges <= self.lidar_range_max[()]]) & np.array([ self.lidar_ranges >= self.lidar_range_min[()]])).reshape(self.data.shape[0], -1)
            self.length = len(self.lidar_stamps)
    def getOneLidarDataAfterMask(self, index):
        tmpData = self.data[index].T
        return tmpData[self.mask[index]]
    def convertFromLaserFrameToBodyFrame3D(self, oneLaserDataInLaserFrame):
        return oneLaserDataInLaserFrame + self.lidarPosToRobot
    def convertFromLaserFrameToBodyFrame2D(self, oneLaserDataInLaserFrame):
        return oneLaserDataInLaserFrame + [self.lidarPosToRobot[0],self.lidarPosToRobot[2]]