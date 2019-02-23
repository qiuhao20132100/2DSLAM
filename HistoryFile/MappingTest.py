import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
import os
from LaserDataLoader import LaserData
from matplotlib import collections  as mc

dataset = 20
address = "data" + os.path.sep;

def convertFromLaserFrameToBodyFrame(lidarPosToRobot, oneLaserDataInLaserFrame):
    return oneLaserDataInLaserFrame + lidarPosToRobot

def show_lidar():
    with np.load(address + "Hokuyo20.npz") as data:
        lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
        lidar_range_min = data["range_min"]  # minimum range value [m]
        lidar_range_max = data["range_max"]  # maximum range value [m]
        lidar_ranges = data["ranges"].T  # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
        angles = np.arange(lidar_angle_min[()], lidar_angle_max[()] + lidar_angle_increment[0][0] * 0.1, lidar_angle_increment[0][0], dtype = np.float64)

        mask = np.array([lidar_ranges[0] >= lidar_range_min]) & np.array([lidar_ranges[0] <= lidar_range_max])
        mask = mask.reshape(1081)
        ranges = lidar_ranges[0][mask]
        angles = angles[mask]
        ax = plt.subplot(1,1,1, projection='polar')
        ax.plot(angles, ranges)
        ax.set_rmax(15)
        ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
        ax.set_rlabel_position(-22.5) # get radial labels away from plotted line
        ax.grid(True)
        ax.set_title("Lidar scan data", va='bottom')
        plt.show()

if __name__ == '__main__':
    ld =LaserData(address + "Hokuyo20.npz")
    for i in range(1):#range(ld.length):
        oneLaserDataInLaserFrame = ld.getOneLidarDataAfterMask(i)

        # lidarPosToRobot = np.array([0, 514.35 / 10000, (298.33 - 330.20 / 2) / 10000])
        # oneLaserDataInBodyFrame = convertFromLaserFrameToBodyFrame(lidarPosToRobot,oneLaserDataInLaserFrame);
        # print(np.vstack((oneLaserDataInLaserFrame[:,0], oneLaserDataInLaserFrame[:,1])).shape)
        # print(oneLaserDataInBodyFrame[100:120])
        # print(oneLaserDataInBodyFrame.shape)
        # fun = lambda x: [[x[0], x[2]], [0,0]]
        # lines = np.array(list(map(lambda x: np.array([[x[0], x[2]], [0,0]]), oneLaserDataInBodyFrame)))
        # #lines = np.array([[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]])
        # print(lines)
        # lc = mc.LineCollection(lines,  linewidths=1)
        # fig, ax = plt.subplots()
        # ax.add_collection(lc)
        # ax.set_xlim([-15, 15])
        # ax.set_ylim([-15, 15])
        # ax.margins(0.1)
        # print("pass")
        # show_lidar()
        # print("pass")