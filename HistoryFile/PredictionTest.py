from OdometryDataLoader import OdometryData
from IMUDataLoader import IMUData
import numpy as np
import matplotlib.pyplot as plt
import math
import os
address = "data" + os.path.sep
encoderdata = OdometryData(address + "Encoders20.npz")
IMUdata = IMUData(address + "Imu20.npz")
road = []
prePos = [0.0, 0.0, 0.0]
road.append([0.0, 0.0])
preTime = encoderdata.encoder_stamps[0]

for i in range(encoderdata.length)[2:]:
    yaw = IMUdata.getOneYawDataByTime(encoderdata.encoder_stamps[i])
    dis = ((encoderdata.encoder_counts[i][0] + encoderdata.encoder_counts[i][2]) / 2.0 * 0.0022 + \
          (encoderdata.encoder_counts[i][1] + encoderdata.encoder_counts[i][3]) / 2.0 * 0.0022) / 2.0
    delta_t = encoderdata.encoder_stamps[i] - preTime
    if (yaw * delta_t == 0):
        continue

    prePos[0] = prePos[0] + dis * \
                (math.sin(yaw * delta_t / 2.0) /(yaw * delta_t / 2.0)) * \
                (math.cos(prePos[2] + yaw * delta_t / 2.0))

    prePos[1] = prePos[1] + dis * \
                (math.sin(yaw * delta_t / 2.0) /(yaw * delta_t / 2.0)) * \
                (math.sin(prePos[2] + yaw * delta_t / 2.0))

    prePos[2] = prePos[2] + yaw * delta_t

    preTime = encoderdata.encoder_stamps[i]
    road.append([prePos[0], prePos[1]])

road = np.array(road)
road = road.T
plt.scatter(road[0,:],road[1,:], marker='o')
plt.show()



