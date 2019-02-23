import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os
import cv2
from IMUDataLoader import IMUData
from LaserDataLoader import  LaserData
from OdometryDataLoader import OdometryData
from KinectDataLoader import KinectData
import math
import random

prefix = "data" + os.path.sep;

def getTransFrombodyToWorld3D(particlePosInWorldFrame):
    particle3DPos = np.array([0, 0, 0])
    particle3DPos[0] = particlePosInWorldFrame[1]
    particle3DPos[1] = -particlePosInWorldFrame[0]
    particle3DPos[2] = 0.127
    particle3DPos = particle3DPos.reshape((3, 1))
    particlesOri = particlePosInWorldFrame[2]

    rotationBody2World = np.array([[np.cos(particlesOri), -np.sin(particlesOri), 0],
                                   [np.sin(particlesOri), np.cos(particlesOri), 0],
                                   [0, 0, 1]])
    transformBody2World = np.hstack((rotationBody2World, particle3DPos))
    transformBody2World = np.vstack((transformBody2World, np.array([[0, 0, 0, 1]])))
    return transformBody2World

# def bodyToWorldOnePoint(dataInBodyFrame, particlePosInWorldFrame):
#     # print("aaaa")
#     # print(particlePosInWorldFrame[0])
#     # print(particlePosInWorldFrame[1])
#     delta_Degree = particlePosInWorldFrame[2]
#     rotate = np.array([[math.cos(delta_Degree), -math.sin(delta_Degree)],[math.sin(delta_Degree), math.cos(delta_Degree)]])
#     # print("good rotate")
#     # print(rotate)
#     dataAfterRotation = rotate.dot(dataInBodyFrame)
#     dataInWordFrame = (dataAfterRotation.T + np.array([particlePosInWorldFrame[0], particlePosInWorldFrame[1]])).T
#     return dataInWordFrame[0], dataInWordFrame[1]

def bodyToWorld(dataInBodyFrame, particlePosInWorldFrame):
    '''
    :param dataInBodyFrame: shape is numOfValidLidarData * 3 (N, 3)
    :param particlePosInWorldFrame: [100 * [x, y, theta]] (100,3)
    :return:(2,N,100)
    '''
    delta_Degree = particlePosInWorldFrame[:,2]
    # rotate = np.zeros((particlePosInWorldFrame.shape[0], 2, 2))
    # rotate[:, 0, 0] = np.cos(delta_Degree)
    # rotate[:, 0, 1] = -np.sin(delta_Degree)
    # rotate[:, 1, 0] = np.sin(delta_Degree)
    # rotate[:, 1, 1] = np.cos(delta_Degree)
    rotate = np.array([[np.cos(delta_Degree),np.sin(delta_Degree)],[-np.sin(delta_Degree),np.cos(delta_Degree)]]).T
    dataAfterRotation = np.dot(rotate, dataInBodyFrame[:,0:2].T) #(100, 2, N)
    dataInWordFrame = (dataAfterRotation.T + np.stack((particlePosInWorldFrame[:,0], particlePosInWorldFrame[:,1])))
    return dataInWordFrame #(N, 2, 100)

def initMap():
    MAP = {}# init MAP
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -30  # meters
    MAP['ymin'] = -30
    MAP['xmax'] = 30
    MAP['ymax'] = 30
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))# cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['binaryMap'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)# DATA TYPE: char or int8
    MAP['logMap'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64)
    MAP['colorMap'] = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
    return MAP

def mapping(particlePosInPhy, scanXPosInPhyInBodyFrame, scanYPosInPhyInBodyFrame, MAP):

    scanXYPosInPhyInWorldFrame = bodyToWorld(np.array([scanXPosInPhyInBodyFrame,scanYPosInPhyInBodyFrame]).T, particlePosInPhy.reshape(1,3))
    scanXPosInPhyInWorldFrame = scanXYPosInPhyInWorldFrame[:, 0, :].T.reshape(-1)
    scanYPosInPhyInWorldFrame = scanXYPosInPhyInWorldFrame[:, 1, :].T.reshape(-1)
    particleAndScanXPosInPhy = np.concatenate([scanXPosInPhyInWorldFrame, [particlePosInPhy[0]]])
    particleAndScanYPosInPhy = np.concatenate([scanYPosInPhyInWorldFrame, [particlePosInPhy[1]]])

    particleAndScanXPosInGrid = (np.ceil((particleAndScanXPosInPhy - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
    particleAndScanYPosInGrid = (np.ceil((particleAndScanYPosInPhy - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)
    particleAndScanXPosInGrid[particleAndScanXPosInGrid >= MAP['sizex']] = MAP['sizex'] - 1
    particleAndScanXPosInGrid[particleAndScanXPosInGrid < 0] = 0
    particleAndScanYPosInGrid[particleAndScanYPosInGrid >= MAP['sizey']] = MAP['sizey'] - 1
    particleAndScanYPosInGrid[particleAndScanYPosInGrid < 0] = 0

    MAP['logMap'][particleAndScanXPosInGrid[0:len(particleAndScanXPosInGrid - 1)], particleAndScanYPosInGrid[0:len(particleAndScanYPosInGrid - 1)]] += 2 * np.log(4)
    polygon = np.zeros((MAP['sizey'], MAP['sizex']))

    occupied_ind = np.vstack((particleAndScanYPosInGrid, particleAndScanXPosInGrid)).T

    ctr = np.array(occupied_ind).reshape((1, -1, 2)).astype(np.int32)
    cv2.drawContours(image = polygon, contours = ctr, contourIdx = 0, color = np.log(0.25), thickness = -1)
    MAP['logMap'] += polygon

def update(scanInPhyInBodyFrame, particles, x_range, y_range, weight, MAP):
    psave = np.array(particles)
    weightSave = np.array(weight)

    #particles shape : n x 3
    scanXYPosInPhyInWorldFrame = bodyToWorld(scanInPhyInBodyFrame, particles)
    #(N, 2, 100)
    scanXYPosInGridInWorldFrame = np.ceil((scanXYPosInPhyInWorldFrame.T - np.array([MAP['xmin'], MAP['ymin']]).reshape(1, 2, 1)) / MAP['res']).astype(np.int32) - 1
    correlationMatrix = map_correlation_mat_version(scanXYPosInGridInWorldFrame, MAP, np.ceil(x_range / MAP['res']).astype(np.int32), np.ceil(y_range / MAP['res']).astype(np.int32))# (100, 81)
    corr_maxs = np.max(correlationMatrix, axis=1)
    corr_maxs_ind = np.argmax(correlationMatrix, axis=1)

    corr_X_update = x_range[corr_maxs_ind % x_range.shape[0]]
    corr_Y_update = y_range[corr_maxs_ind // y_range.shape[0]]
    particles[:,0] = corr_X_update + particles[:,0]
    particles[:,1] = corr_Y_update + particles[:,1]

    corr_maxmax = corr_maxs[np.argmax(corr_maxs)]
    weightSum = np.log(weight) + corr_maxs - corr_maxmax
    normalize = np.log(np.sum(np.exp(weightSum)))
    weightSum = weightSum - normalize

    wei_update = np.exp(weightSum)
    ind_target = wei_update.argmax()
    return particles, wei_update, ind_target
    # return psave, weightSave, 0

def motionModel(dis, delta_t, yaw, particles):
    particleOri = particles[:,2]
    a = - dis * \
            (np.sin(yaw * delta_t / 2.0) / (yaw * delta_t / 2.0)) * \
            (np.sin(particleOri + yaw * delta_t / 2.0))
    b = dis * \
            (np.sin(yaw * delta_t / 2.0) / (yaw * delta_t / 2.0)) * \
            (np.cos(particleOri + yaw * delta_t / 2.0))
    c = yaw * delta_t * np.ones((len(particleOri)))

    posUpdate = np.stack((a,b,c), axis = 1)

    # posUpdate = np.zeros(particles.shape)
    # posUpdate[:,0] = - dis * \
    #             (np.sin(yaw * delta_t / 2.0) /(yaw * delta_t / 2.0)) * \
    #             (np.sin(particleOri + yaw * delta_t / 2.0))
    #
    # posUpdate[:,1] = dis * \
    #             (np.sin(yaw * delta_t / 2.0) /(yaw * delta_t / 2.0)) * \
    #             (np.cos(particleOri + yaw * delta_t / 2.0))
    #
    # posUpdate[:,2] = yaw * delta_t  * np.ones((len(particleOri)))
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

def mapRGBToColorMap(MAP, rgb, positionWorld):
    # transfer the data into grid
    positionWorldXY = positionWorld[:, :2]
    tmp = np.array(positionWorldXY[:, 0])
    positionWorldXY[:, 0] = -positionWorldXY[:, 1]
    positionWorldXY[:, 1] = tmp
    pixelPosInGrid = np.ceil(
        (positionWorldXY.T - np.array([MAP['xmin'], MAP['ymin']]).reshape(2, 1)) / MAP['res']).astype(np.int32) - 1
    pixelPosInGrid[0, :][pixelPosInGrid[0, :] < 0] = 0
    pixelPosInGrid[0, :][pixelPosInGrid[0, :] >= MAP['colorMap'].shape[0]] = MAP['colorMap'].shape[0] - 1
    pixelPosInGrid[1, :][pixelPosInGrid[1, :] < 0] = 0
    pixelPosInGrid[1, :][pixelPosInGrid[1, :] >= MAP['colorMap'].shape[1]] = MAP['colorMap'].shape[1] - 1

    MAP['colorMap'][pixelPosInGrid[0, :], pixelPosInGrid[1, :], :] = rgb[rgbj, rgbi, :]
    return

def drawMap(map, route, particleHistory, index):
    posMask = np.array(map > 0)
    negMask = np.array(map < 0)
    img = np.zeros(map.shape, dtype = np.uint8)
    img[:,:] = map[:,:]
    img[posMask] = 250
    img[negMask] = 30
    for point in route:
        if (point[0] >= 0 and point[0] < map.shape[0] and point[1] >= 0 and point[1] < map.shape[1]):
            img[point[0],point[1]] = 50
    for row in range(particleHistory.shape[0]):
        img[particleHistory[row][0], particleHistory[row][1]] = 70
    plt.imsave(("picture" + os.path.sep + str(index) + "map.png"),img)
    # cv2.imshow('img',img)
    # cv2.waitKey()

def map_correlation_mat_version( vp, MAP, xs=np.arange(-4, 5), ys=np.arange(-4, 5)):
    """
    :param vp: 2*1081*100, represent xy, angle, state_n
    :param xs: x shift, a numpy array, arange from left shift to right shift np.arange(-4, 5)
    :param ys: y shift, a numpy array, arange from left shift to right shift np.arange(-4, 5)
    :return: N*81 np array.
    """
    vp = np.swapaxes(vp, 0, 1)
    vp = np.swapaxes(vp, 1, 2)
    x_vp = vp[0, :, :]
    y_vp = vp[1, :, :]

    x_vp_shape = x_vp.shape  # (1081, 100)
    x_vp = x_vp.reshape(-1, 1)  # prior n particle, then 1081 angle
    y_vp = y_vp.reshape(-1, 1)  # prior n particle, then 1081 angle

    y_change, x_change = np.where(np.ones((ys.shape[0], xs.shape[0])) == 1)
    x_change = (x_change + xs[0]).astype(np.int32)
    y_change = (y_change + ys[0]).astype(np.int32)
    # print(np.max(x_vp))
    # print(x_vp.shape)
    x_vp = np.dot(x_vp, np.ones((1, x_change.shape[0]), dtype=np.int32)) + x_change.reshape(1, -1)
    y_vp = np.dot(y_vp, np.ones((1, y_change.shape[0]), dtype=np.int32)) + y_change.reshape(1, -1)
    # print(np.max(x_vp))
    flag = np.logical_and(np.logical_and(x_vp >= 0, x_vp < MAP['sizex']),
                          np.logical_and(y_vp >= 0, y_vp < MAP['sizey']))
    # print(flag.any())
    x_vp[np.logical_not(flag)] = 0
    y_vp[np.logical_not(flag)] = 0

    img_extract = MAP['binaryMap'][x_vp, y_vp]
    img_extract[np.logical_not(flag)] = 0

    img_extract = img_extract.reshape(x_vp_shape[0], x_vp_shape[1], -1)  # (1081, 100, 81)
    # print(img_extract.shape)
    img_extract = np.sum(img_extract, axis=0)  # (100, 81)

    return img_extract

if __name__ == '__main__':

    dataSet = 20
    numOfParticles = 100
    Threshold = 35
    noiseFactor = np.array([1, 1, 10])
    needTexture = False
    height_threshold = [-2, 0.25]

    #set parameters
    particles = np.zeros((numOfParticles, 3))
    weight = 1.0 / numOfParticles * np.ones((numOfParticles), dtype = np.float64)
    yawData = IMUData(prefix + "Imu"+ str(dataSet) + ".npz")
    encoderData = OdometryData(prefix + "Encoders"+ str(dataSet) + ".npz")
    scanData = LaserData(prefix + "Hokuyo"+ str(dataSet) + ".npz")
    if needTexture:
        rgbdData = KinectData(prefix + "dataRGBD" + os.path.sep, dataSet)
    MAP = initMap()

    xPhy = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
    yPhy = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # z-positions of each pixel of the map
    x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)

    #initial map by first scan
    initialScanData = scanData.convertFromLaserFrameToBodyFrame3D(scanData.getOneLidarDataAfterMask(0))
    mapping(particles[0,:], initialScanData[:,0], initialScanData[:,1], MAP)

    #set start index and time
    time = scanData.lidar_stamps[0]
    indexOfEncoderStartPoint = 0
    while (indexOfEncoderStartPoint < encoderData.length and encoderData.encoder_stamps[indexOfEncoderStartPoint] < time):
        indexOfEncoderStartPoint += 1
    pre_encoder_time = time

    #route variables initialization
    route = []
    particlesHistory = np.zeros((1,2), dtype = np.int32)

    #main loop
    for indexOfEncoder in range(indexOfEncoderStartPoint, encoderData.length):

        #print process
        if (indexOfEncoder % 50 == 0):
            print("%3.2f" % (((indexOfEncoder + 1.0) * 100 / encoderData.length)) + '% completed')
        if (indexOfEncoder % 100 == 0):
            drawMap(MAP['logMap'], route, particlesHistory, indexOfEncoder / 100)
        # if (indexOfEncoder % 1000 == 0):
        #     break

        # prediction
        time = encoderData.encoder_stamps[indexOfEncoder]
        yaw = yawData.getOneYawDataByTime(time)
        dis = ((encoderData.encoder_counts[indexOfEncoder][0] + encoderData.encoder_counts[indexOfEncoder][2]) / 2.0 * 0.0022 + \
               (encoderData.encoder_counts[indexOfEncoder][1] + encoderData.encoder_counts[indexOfEncoder][3]) / 2.0 * 0.0022) / 2.0
        delta_t = time - pre_encoder_time
        posUpdate = motionModel(dis, delta_t, yaw, particles)
        noise = np.einsum('..., ...', noiseFactor, np.random.normal(0, 1e-3, (numOfParticles, 1)))
        particles = particles + posUpdate + noise
        pre_encoder_time = time

        # update step
        # bestParticleIndex = 0
        currScanData = scanData.convertFromLaserFrameToBodyFrame3D(scanData.getOneLidarDataAfterMaskByTime(time))
        MAP['binaryMap'] = np.zeros(MAP['logMap'].shape)
        MAP['binaryMap'][MAP['logMap'] > 0] = 1
        particles, weight, bestParticleIndex = update(currScanData, particles,x_range, y_range, weight, MAP)


        # mapping
        mapping(particles[bestParticleIndex], currScanData[:,0], currScanData[:,1], MAP)

        #print map
        routeX = (np.ceil((particles[bestParticleIndex][0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        routeY = (np.ceil((particles[bestParticleIndex][1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)
        route.append([routeX, routeY])
        # if loopCounter % 100 == 0:
        #     print("round:" + str(loopCounter))
        #     # if (loopCounter >= 500):
        #     #     break
        #     particlesX = (np.ceil((particles[:,0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        #     particlesY = (np.ceil((particles[:,1] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        #     particlesXY = np.stack((particlesX, particlesY)).T
        #     particlesHistory = np.concatenate((particlesHistory,particlesXY))


        #texture

        if needTexture:
            #get data
            rgb = rgbdData.getOneRGBDataByTime(time)
            disparity = rgbdData.getOneDisparityDataByTime(time)
            #calculate rgbIndex and depth
            depth, rgbi, rgbj = rgbdData.getDepthAndIndex(rgb, disparity)
            #transfer from pixel position to camera position
            pixels = np.vstack((rgbi.reshape(1, -1), rgbj.reshape(1, -1), np.ones_like(rgbi.reshape(1, -1))))
            positionCamera = rgbdData.convertPixelToBody(pixels, depth)
            #transfer from camera to world
            #the transfer between camera and body is fixed
            #we only need to calculate the transfer between body and world
            transformBody2World = getTransFrombodyToWorld3D(particles[bestParticleIndex])
            positionWorld = np.dot(np.dot(transformBody2World,rgbdData.transfromCamToBody),positionCamera).T  # num_pixels * 4

            #get mask by height Threshold and filter the data
            pixelOnTheGroundMask = np.logical_and(positionWorld[:, 2] >= height_threshold[0], positionWorld[:, 2] <= height_threshold[1])
            positionWorld = positionWorld[pixelOnTheGroundMask, :]
            rgbi = rgbi[pixelOnTheGroundMask]
            rgbj = rgbj[pixelOnTheGroundMask]
            #draw
            mapRGBToColorMap(MAP, rgb, positionWorld)

        # resample particles if necessary
        N_eff = 1 / np.sum(np.square(weight))
        if N_eff <Threshold:
            particles = resample(numOfParticles, weight, particles)
            weight = 1.0 / numOfParticles * np.ones((numOfParticles), dtype=np.float64)

    drawMap(MAP['logMap'],route, particlesHistory, "final")
    plt.imsave("colorMap.png", MAP['colorMap'])
    cv2.imshow('wtf',MAP['colorMap'])
    cv2.waitKey()

