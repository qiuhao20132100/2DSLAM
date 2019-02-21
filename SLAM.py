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
    MAP['xmin'] = -25  # meters
    MAP['ymin'] = -25
    MAP['xmax'] = 25
    MAP['ymax'] = 25
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))# cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['binaryMap'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)# DATA TYPE: char or int8
    MAP['logMap'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64)
    return MAP

def drawMap(map, route, particleHistory):
    posMask = np.array(map > 0)
    negMask = np.array(map < 0)
    img = np.zeros(map.shape, dtype = np.uint8)
    img[posMask] = 10
    img[negMask] = 30
    for point in route:
        if (point[0] >= 0 and point[0] < map.shape[0] and point[1] >= 0 and point[1] < map.shape[1]):
            img[point[0],point[1]] = 50
    for row in range(particleHistory.shape[0]):
        img[particleHistory[row][0], particleHistory[row][1]] = 70
    plt.imshow(img)
    plt.imsave("map.png",img)
    plt.show()
    plt.waitforbuttonpress()

def mapping(particlePosInPhy, scanXPosInPhyInBodyFrame, scanYPosInPhyInBodyFrame, MAP):
    scanXPosInPhyInWorldFrame, scanYPosInPhyInWorldFrame = bodyToWorld(np.array([scanXPosInPhyInBodyFrame,scanYPosInPhyInBodyFrame]), particlePosInPhy)

    particleAndScanXPosInPhy = np.concatenate([scanXPosInPhyInWorldFrame, [particlePosInPhy[0]]])
    particleAndScanYPosInPhy = np.concatenate([scanYPosInPhyInWorldFrame, [particlePosInPhy[1]]])
    particleAndScanXPosInGrid = (np.ceil((particleAndScanXPosInPhy - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
    particleAndScanYPosInGrid = (np.ceil((particleAndScanYPosInPhy - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)
    particleAndScanXPosInGrid[particleAndScanXPosInGrid >= MAP['sizex']] = MAP['sizex'] - 1
    particleAndScanXPosInGrid[particleAndScanXPosInGrid < 0] = 0
    particleAndScanYPosInGrid[particleAndScanYPosInGrid >= MAP['sizex']] = MAP['sizex'] - 1
    particleAndScanYPosInGrid[particleAndScanYPosInGrid < 0] = 0

    MAP['logMap'][particleAndScanXPosInGrid[0:len(particleAndScanXPosInGrid - 1)], particleAndScanYPosInGrid[0:len(particleAndScanYPosInGrid - 1)]] += 2 * np.log(4)
    polygon = np.zeros((MAP['sizey'], MAP['sizex']))

    occupied_ind = np.vstack((particleAndScanYPosInGrid, particleAndScanXPosInGrid)).T

    ctr = np.array(occupied_ind).reshape((1, -1, 2)).astype(np.int32)
    cv2.drawContours(image = polygon, contours = ctr, contourIdx = 0, color = np.log(0.25), thickness = -1)
    MAP['logMap'] += polygon

def update(numOfParticles, scanInPhyInBodyFrame, particles, xim, yim, x_range, y_range, weight, binaryMap):

    corr = np.zeros((numOfParticles))
    range_size = len(x_range)
    for i in range(numOfParticles):
        scanXPosInPhyInWorldFrame, scanYPosInPhyInWorldFrame = bodyToWorld(np.array([scanInPhyInBodyFrame[:,0], scanInPhyInBodyFrame[:,1]]), particles[i])
        Y = np.stack((scanXPosInPhyInWorldFrame, scanYPosInPhyInWorldFrame))
        correlation = mapCorrelation(binaryMap, xim, yim, Y, x_range, y_range)
        ind = np.argmax(correlation)
        corr[i] = correlation[ind // range_size][ind % range_size]
        # particles[i, 0] += x_range[ind // range_size]
        # particles[i, 1] += y_range[ind % range_size]

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

    dataSet = 21
    numOfParticles = 100
    Threshold = 35
    noiseFactor = np.array([10, 10, 1])


    #set parameters
    particles = np.zeros((numOfParticles, 3))
    weight = 1.0 / numOfParticles * np.ones((numOfParticles), dtype = np.float64)
    yawData = IMUData(prefix + "Imu"+ str(dataSet) + ".npz")
    encoderData = OdometryData(prefix + "Encoders"+ str(dataSet) + ".npz")
    scanData = LaserData(prefix + "Hokuyo"+ str(dataSet) + ".npz")
    MAP = initMap()

    xPhy = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
    yPhy = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # z-positions of each pixel of the map
    x_range = np.arange(-0.05, 0.05 + 0.05, 0.05)
    y_range = np.arange(-0.05, 0.05 + 0.05, 0.05)

    #initial
    initialScanData = scanData.convertFromLaserFrameToBodyFrame3D(scanData.getOneLidarDataAfterMask(0))
    mapping(particles[0,:], initialScanData[:,0], initialScanData[:,1], MAP)
    time = scanData.lidar_stamps[0]
    indexOfScan = 1
    indexOfEncoder = 0
    pre_encoder_time = time
    route = []
    particlesHistory = np.zeros((1,2), dtype = np.int32)

    encoderCounter = 0
    while indexOfScan < scanData.length and indexOfEncoder < encoderData.length:
        # print("looping")
        while (indexOfEncoder < encoderData.length and encoderData.encoder_stamps[indexOfEncoder] < time):
            indexOfEncoder += 1
        if (indexOfEncoder == encoderData.length):
            break
        # prediction
        encoderCounter += 1

        time = encoderData.encoder_stamps[indexOfEncoder]
        yaw = yawData.getOneYawDataByTime(time)
        dis = ((encoderData.encoder_counts[indexOfEncoder][0] + encoderData.encoder_counts[indexOfEncoder][2]) / 2.0 * 0.0022 + \
               (encoderData.encoder_counts[indexOfEncoder][1] + encoderData.encoder_counts[indexOfEncoder][3]) / 2.0 * 0.0022) / 2.0
        delta_t = time - pre_encoder_time
        posUpdate = motionModel(dis, delta_t, yaw, particles)
        noise = np.einsum('..., ...', noiseFactor, np.random.normal(0, 1e-3, (numOfParticles, 1)))
        particles = particles + posUpdate + noise
        # plt.scatter(particles[:,0], particles[:,1])
        pre_encoder_time = time

        while(indexOfScan < scanData.length and scanData.lidar_stamps[indexOfScan] < time):
            indexOfScan += 1
        if (indexOfScan == scanData.length):
            break

        time = scanData.lidar_stamps[indexOfScan]
        # update step
        currScanData = scanData.convertFromLaserFrameToBodyFrame3D(scanData.getOneLidarDataAfterMask(indexOfScan))
        MAP['binaryMap'] = np.zeros(MAP['logMap'].shape)
        MAP['binaryMap'][MAP['logMap'] > 0] = 1
        # bestParticleIndex = 0
        particles, weight, bestParticleIndex = update(numOfParticles, currScanData, particles, xPhy, yPhy, x_range, y_range, weight, MAP['binaryMap'])
        # mapping
        mapping(particles[bestParticleIndex], currScanData[:,0], currScanData[:,1], MAP)

        #print map
        routeX = (np.ceil((particles[bestParticleIndex][0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        routeY = (np.ceil((particles[bestParticleIndex][1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)
        route.append([routeX, routeY])
        if encoderCounter % 100 == 0:
            particlesX = (np.ceil((particles[:,0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
            particlesY = (np.ceil((particles[:,1] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
            particlesXY = np.stack((particlesX, particlesY)).T
            particlesHistory = np.concatenate((particlesHistory,particlesXY))
            print(particles)
            print(weight)


        # resample particles if necessary
        N_eff = 1 / np.sum(np.square(weight))
        if N_eff <Threshold:
            particles = resample(numOfParticles, weight, particles)
            weight = 1.0 / numOfParticles * np.ones((numOfParticles), dtype=np.float64)


    # plt.show()
    # plt.waitforbuttonpress()
    drawMap(MAP['logMap'],route, particlesHistory)
