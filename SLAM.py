import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
import os
import cv2
from IMUDataLoader import IMUData
from LaserDataLoader import  LaserData
from OdometryDataLoader import OdometryData
import math
import random
from map_utils import mapCorrelation
prefix = "data" + os.path.sep;

def bodyToWorld(dataInBodyFrame, particlePosInWorldFrame):
    delta_Degree = particlePosInWorldFrame[2]
    rotate = np.array([[math.cos(delta_Degree), -math.sin(delta_Degree)],[math.sin(delta_Degree), math.cos(delta_Degree)]])
    dataAfterRotation = rotate.dot(dataInBodyFrame)
    dataInWordFrame = (dataAfterRotation.T + np.array([particlePosInWorldFrame[0], particlePosInWorldFrame[1]])).T
    return dataInWordFrame[0],dataInWordFrame[1]

def initMap():
    MAP = {}# init MAP
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -40  # meters
    MAP['ymin'] = -40
    MAP['xmax'] = 40
    MAP['ymax'] = 40
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))# cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)# DATA TYPE: char or int8
    MAP['logMap'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64)
    return MAP

def drawMap(map, route):

    posMask = np.array(map > 0)
    negMask = np.array(map < 0)
    img = np.zeros(map.shape, dtype = np.uint8)
    print(posMask.shape)
    print(img.shape)
    img[posMask] = 10
    img[negMask] = 120
    for point in route:
        img[point[0],point[1]] = 250
    plt.imshow(img)
    plt.show()
    plt.waitforbuttonpress()

def drawLogMap(map):
    posMask = np.array(map > 0)
    negMask = np.array(map < 0)
    img = np.zeros(map.shape, dtype = np.int8)
    print(posMask.shape)
    print(img.shape)
    img[posMask] = 10
    img[negMask] = 250
    plt.imshow(img)
    plt.show()
    plt.waitforbuttonpress()

def mapping(particlePosInPhy, scanXPosInPhyInBodyFrame, scanYPosInPhyInBodyFrame, MAP):
    scanXPosInPhyInWorldFrame, scanYPosInPhyInWorldFrame = bodyToWorld(np.array([scanXPosInPhyInBodyFrame,scanYPosInPhyInBodyFrame]), particlePosInPhy)

    particleAndScanXPosInPhy = np.concatenate([scanXPosInPhyInWorldFrame, [particlePosInPhy[0]]])
    particleAndScanYPosInPhy = np.concatenate([scanYPosInPhyInWorldFrame, [particlePosInPhy[1]]])
    particleAndScanXPosInGrid = (np.ceil((particleAndScanXPosInPhy - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
    particleAndScanYPosInGrid = (np.ceil((particleAndScanYPosInPhy - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)
    particleAndScanXPosInGrid[particleAndScanXPosInGrid > 1600] = 1600
    particleAndScanXPosInGrid[particleAndScanXPosInGrid < 0] = 0
    particleAndScanYPosInGrid[particleAndScanYPosInGrid > 1600] = 1600
    particleAndScanYPosInGrid[particleAndScanYPosInGrid < 0] = 0

    MAP['logMap'][particleAndScanXPosInGrid[0:len(particleAndScanXPosInGrid - 1)], particleAndScanYPosInGrid[0:len(particleAndScanYPosInGrid - 1)]] += 2 * np.log(4)
    polygon = np.zeros((MAP['sizey'], MAP['sizex']))

    occupied_ind = np.vstack((particleAndScanYPosInGrid, particleAndScanXPosInGrid)).T

    ctr = np.array(occupied_ind).reshape((1, -1, 2)).astype(np.int32)
    cv2.drawContours(image = polygon, contours = ctr, contourIdx = 0, color = np.log(0.25), thickness = -1)
    MAP['logMap'] += polygon
    # drawLogMap(MAP['logMap'])
    # print("pass")

def update(numOfParticles, scanInPhyInBodyFrame, particles, xim, yim, x_range, y_range, weight, binaryMap):

    corr = np.zeros((numOfParticles))
    range_size = len(x_range)
    for i in range(numOfParticles):
        scanXPosInPhyInWorldFrame, scanYPosInPhyInWorldFrame = bodyToWorld(np.array([scanInPhyInBodyFrame[:,0], scanInPhyInBodyFrame[:,1]]), particles[i])
        Y = np.stack((scanXPosInPhyInWorldFrame, scanYPosInPhyInWorldFrame))
        correlation = mapCorrelation(binaryMap, xim, yim, Y, x_range, y_range)
        ind = np.argmax(correlation)
        corr[i] = correlation[ind // range_size][ind % range_size]
        particles[i, 0] += x_range[ind // range_size]
        particles[i, 1] += y_range[ind % range_size]

    corr_max = corr[np.argmax(corr)]
    weightSum = np.log(weight) + corr - corr_max
    normalize = np.log(np.sum(np.exp(weightSum)))
    weightSum = weightSum - normalize

    wei_update = np.exp(weightSum)
    ind_target = wei_update.argmax()
    return particles, wei_update, ind_target

def motionModel(dis, delta_t, yaw, particles):
    particleOri = particles[:,2]
    posUpdate = np.zeros(particles.shape)

    posUpdate[:,0] = - dis * \
                (np.sin(yaw * delta_t / 2.0) /(yaw * delta_t / 2.0)) * \
                (np.sin(particleOri + yaw * delta_t / 2.0))

    posUpdate[:,1] = dis * \
                (np.sin(yaw * delta_t / 2.0) /(yaw * delta_t / 2.0)) * \
                (np.cos(particleOri + yaw * delta_t / 2.0))

    posUpdate[:,2] = yaw * delta_t  * np.ones((len(particleOri)))
    return posUpdate

def resample(N, weight, particles):
    particle_New = np.zeros((N, 3))
    r = random.uniform(0, 1.0 / N)

    c, i = weight[0], 0
    for m in range(N):
        u = r + m * (1.0 / N)

        while u > c:
            i = i + 1
            c = c + weight[i]

        particle_New[m, :] = particles[i, :]

    return particle_New

if __name__ == '__main__':
    numOfParticles = 4
    Threshold = 2
    particles = np.zeros((numOfParticles, 3))
    weight = 1.0 / numOfParticles * np.ones((numOfParticles), dtype = np.float64)
    yawData = IMUData(prefix + "Imu20.npz")
    encoderData = OdometryData(prefix + "Encoders20.npz")
    scanData = LaserData(prefix + "Hokuyo20.npz")

    MAP = initMap()

    xPhy = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
    yPhy = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # z-positions of each pixel of the map
    x_range = np.arange(-0.05, 0.06, 0.05)
    y_range = np.arange(-0.05, 0.06, 0.05)

    #initial
    initalScanData = scanData.convertFromLaserFrameToBodyFrame3D(scanData.getOneLidarDataAfterMask(0))
    # print(initalScanData)
    mapping(particles[0,:], initalScanData[:,0], initalScanData[:,1], MAP)
    time = scanData.lidar_stamps[0]
    indexOfScan = 1
    indexOfEncoder = 0
    pre_encoder_time = time
    noiseFactor = np.array([1, 1, 10])
    route = []

    encoderCounter = 0
    while indexOfScan < scanData.length and indexOfEncoder < encoderData.length:
        # print("looping")
        while (indexOfEncoder < encoderData.length and encoderData.encoder_stamps[indexOfEncoder] < time):
            indexOfEncoder += 1
        if (indexOfEncoder == encoderData.length):
            break
        # prediction
        # encoderCounter += 1
        # if (encoderCounter > 500):
        #     break
        time = encoderData.encoder_stamps[indexOfEncoder]
        yaw = yawData.getOneYawDataByTime(time)
        dis = ((encoderData.encoder_counts[indexOfEncoder][0] + encoderData.encoder_counts[indexOfEncoder][2]) / 2.0 * 0.0022 + \
               (encoderData.encoder_counts[indexOfEncoder][1] + encoderData.encoder_counts[indexOfEncoder][3]) / 2.0 * 0.0022) / 2.0
        delta_t = time - pre_encoder_time
        posUpdate = motionModel(dis, delta_t, yaw, particles)
        noise = np.einsum('..., ...', noiseFactor, np.random.normal(0, 1e-3, (numOfParticles, 1)))
        particles = particles + posUpdate # + noise
        # plt.scatter(particles[:,0], particles[:,1])
        pre_encoder_time = time

        while(indexOfScan < scanData.length and scanData.lidar_stamps[indexOfScan] < time):
            indexOfScan += 1
        if (indexOfScan == scanData.length):
            break

        time = scanData.lidar_stamps[indexOfScan]
        # update step
        currScanData = scanData.convertFromLaserFrameToBodyFrame3D(scanData.getOneLidarDataAfterMask(indexOfScan))
        binaryMap = np.zeros(MAP['logMap'].shape)
        binaryMap[MAP['logMap'] > 0] = 1
        # bestParticleIndex = 0
        paarticles, weight, bestParticleIndex = update(numOfParticles, currScanData, particles, xPhy, yPhy, x_range, y_range, weight, binaryMap)
        # mapping
        mapping(particles[bestParticleIndex], currScanData[:,0], currScanData[:,1], MAP)
        routeX = (np.ceil((particles[bestParticleIndex][0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        routeY = (np.ceil((particles[bestParticleIndex][1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)
        route.append([routeX, routeY])
        # resample particles if necessary
        N_eff = 1 / np.sum(np.square(weight))
        if N_eff <Threshold:
            particles = resample(numOfParticles, weight, particles)
            weight = np.einsum('..., ...', 1.0 / numOfParticles, np.ones((numOfParticles, 1)))


    # plt.show()
    # plt.waitforbuttonpress()
    print(encoderCounter)
    drawMap(MAP['logMap'],route)
