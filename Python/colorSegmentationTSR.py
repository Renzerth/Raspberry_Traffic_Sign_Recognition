#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import cv2
import time
import numpy as np
import math
import glob
import pickle
import os

from collections import OrderedDict
from scipy.spatial import distance as dist
from skimage import feature

#import colorcorrect.algorithm as cca
#import matplotlib.pyplot as plt
#from scipy.ndimage.filters import gaussian_filter
#from skimage import img_as_float

from threading import Thread
from imutils.video import FPS

#%% Camara class definition
class liveStreaming:
    def __init__(self, backSource, widthRes=640, heightRes=480, framerate=15):
        # Camera parameters
        self.streaming = cv2.VideoCapture(backSource)
        self.streaming.set(cv2.CAP_PROP_FPS, framerate)
        self.streaming.set(cv2.CAP_PROP_FRAME_WIDTH, widthRes)
        self.streaming.set(cv2.CAP_PROP_FRAME_HEIGHT, heightRes)
        self.streaming.set(cv2.CAP_PROP_FOURCC, 6)

        self.keepReading = True
        pass
    
    def startThread(self):
        # https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
        Thread(target=self.updateStream, args=()).start()
        return self

    def updateStream(self):
        while self.keepReading:
            (self.acquired, self.currentFrame) = self.streaming.read()

        self.streaming.release()
        
    def readFrame(self):
        return self.currentFrame        
    
    def stopRecording(self):
        self.keepReading = False


class colorEnhancing:
    def __init__(self, gamma = 1.3):
        self.gammaTable = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        self.clahe = cv2.createCLAHE(2, (3,3))
        self.clahe.setClipLimit(2)
        pass
    
    def adjust_gamma(self, image):
        # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        return cv2.LUT(image, self.gammaTable)
    
    def balanceChannelRange(self, channel, perc = 5):
        mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
        channel = np.clip((channel-mi)*255.0/(ma-mi), 0, 255).astype(np.uint8)
        return channel
    
    def apply_mask(self,matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()
    
    def apply_threshold(self,matrix, low_value, high_value):
        low_mask = matrix < low_value
        matrix = self.apply_mask(matrix, low_mask, low_value)
        high_mask = matrix > high_value
        matrix = self.apply_mask(matrix, high_mask, high_value)
        return matrix
    
    def simplest_cb(self,img, half_percent = 0.025):
        #https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
        #https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
        out_channels = []
        for colors in range(3):
            channel = img[:,:,colors]
            flat = np.sort(channel.reshape(channel.size))
            low_val  = flat[math.floor(flat.shape[0] * half_percent)]
            high_val = flat[math.ceil(flat.shape[0] * (1.0 - half_percent))]
            thresholded = self.apply_threshold(channel, low_val, high_val)
            out_channels.append(cv2.normalize(thresholded, None, 0, 255, cv2.NORM_MINMAX))
        return cv2.merge(out_channels)
    
    def grayWorldBalance(self,frame):
        balancedImage = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        correctionFactors = np.subtract(cv2.mean(balancedImage[:,:,1:3])[0:2], 128)/ 280.5
        balancedImage[:,:,1] = balancedImage[:,:,1] - balancedImage[:,:,0]*correctionFactors[0]
        balancedImage[:,:,2] = balancedImage[:,:,2] - balancedImage[:,:,0]*correctionFactors[1]
        balancedImage = cv2.cvtColor(balancedImage, cv2.COLOR_LAB2RGB)
        return balancedImage
    
    def equalizeColorlHist(self,frame):
        for channel in range(3):
            frame[:,:,channel] = cv2.equalizeHist(frame[:,:,channel])
        return frame
    
    def equalizeColorCLAHE(self,frame):
        for channel in range(3):
            frame[:,:,channel] = self.clahe.apply(frame[:,:,channel])
        return frame
    
    def equalizeColorSP(self,frame):
        #https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
        SPChannels = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb) # cv2.COLOR_RGB2LAB
        #SPChannels[:,:,0] = self.clahe.apply(SPChannels[:,:,0])
        SPChannels[:,:,0] = self.clahe.apply(SPChannels[:,:,0]) #cv2.equalizeHist(SPChannels[:,:,0])
        equalizedImage = cv2.cvtColor(SPChannels, cv2.COLOR_YCrCb2RGB)
        return equalizedImage
        
    def balanceImage(self,image):
        balancedImage = self.adjust_gamma(image)
        #balancedImage  = np.dstack([self.balanceChannelRange(channel) for channel in cv2.split(balancedImage)])
        #balancedImage = self.simplest_cb(balancedImage)
        #balancedImage = self.equalizeColorlHist(balancedImage)
        #balancedImage = self.equalizeColorCLAHE(balancedImage)
        balancedImage = self.equalizeColorSP(balancedImage)
        #balancedImage = self.grayWorldBalance(balancedImage)
        #cv.xphoto_WhiteBalancer.balanceWhite
        #balancedImage = cca.grey_world(balancedImage)
        return balancedImage


class deconvolutionFilter:
    def __init__(self, imageResolution, kernelSize = 5, angle = np.pi/6):
        self.imageSize = imageResolution
        self.psfKernSize = kernelSize
        self.motionAngle = angle
        self.zSize = 65
        self.noise = 0
        
        self.createKernels().computeKernelSpectra()

    def blur_edge(self,imageToProcess):
        d = self.psfKernSize
        img_pad = cv2.copyMakeBorder(imageToProcess, d, d, d, d, cv2.BORDER_WRAP)
        img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
        y, x = np.indices((self.imageResolution[1], self.imageResolution[0]))
        dist = np.dstack([x, self.imageResolution[0]-x-1, y, self.imageResolution[1]-y-1]).min(-1)
        w = np.minimum(np.float32(dist)/d, 1.0)
        return imageToProcess*w + img_blur*(1-w)

    def computeMotionKernel(self):
        kern = np.ones((1, self.psfKernSize), np.float32)
        c, s = np.cos(self.motionAngle), np.sin(self.motionAngle)
        A = np.float32([[c, -s, 0], [s, c, 0]])
        sz2 = self.zSize // 2
        A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((self.psfKernSize-1)*0.5, 0))
        self.motionKernel = cv2.warpAffine(kern, A, (self.zSize, self.zSize), flags = cv2.INTER_CUBIC)
        return self

    def computeDefocusKernel(self):
        kern = np.zeros((self.zSize, self.zSize), np.uint8)
        cv2.circle(kern, (self.zSize, self.zSize), self.psfKernSize, 255, -1, cv2.LINE_AA, shift=1)
        self.defocusKernel = np.float32(kern) / 255.0
        return self

    def createKernels(self):
        self.computeDefocusKernel()
        self.computeMotionKernel()
        
        self.motionKernel /= self.motionKernel.sum()
        self.defocusKernel /= self.defocusKernel.sum()
        
        psfPadDefocus = np.zeros((self.imageSize[0], self.imageSize[1]),np.float32)
        psfPadMotion = psfPadDefocus
        
        self.psfKernSize
        psfPadDefocus[:self.zSize, :self.zSize] = self.motionKernel
        psfPadMotion[:self.zSize, :self.zSize] = self.defocusKernel
        self.convolutionKernel = [psfPadDefocus, psfPadMotion]
        return self

    def preProcessDFT(self):
        nrows = cv2.getOptimalDFTSize(self.imageSize[0])
        ncols = cv2.getOptimalDFTSize(self.imageSize[1])
        
        bottomPadding = nrows - self.imageSize[0]
        rightPadding = ncols - self.imageSize[1]
        self.paddingRanges = [rightPadding, bottomPadding]
        return self
     
    def computeKernelSpectra(self):
        # https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/ImgProc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.cpp
        # https://docs.opencv.org/ref/master/de/d3c/tutorial_out_of_focus_deblur_filter.html
        # https://github.com/lvxiaoxin/Wiener-filter/blob/master/main.py
        # https://github.com/opencv/opencv/blob/master/samples/python/deconvolution.py
        self.preProcessDFT()
        
        psfOptSizeDefocus = cv2.copyMakeBorder(self.convolutionKernel[0], 0, self.paddingRanges[1], 0, self.paddingRanges[0], cv2.BORDER_CONSTANT, value = 0)
        psfOptSizeMotion = cv2.copyMakeBorder(self.convolutionKernel[1], 0, self.paddingRanges[1], 0, self.paddingRanges[0], cv2.BORDER_CONSTANT, value = 0)
        
        OTFDefocus = cv2.dft(psfOptSizeDefocus, flags = cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = self.zSize)
        OTFMotion = cv2.dft(psfOptSizeMotion, flags = cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = self.zSize)
        
        self.OTF = cv2.mulSpectrums(OTFDefocus,OTFMotion, False)
        self.WignerKernel = self.OTF / ((self.OTF**2).sum(-1) + self.noise)[...,np.newaxis]
        return self
    
    def deconvoluteImage(self, imageToProcess):
        imageOptPadding = cv2.copyMakeBorder(imageToProcess, 0, self.paddingRanges[1], 0, self.paddingRanges[0], cv2.BORDER_CONSTANT, value = 0)
        imageSpectrum = cv2.dft(np.float32(imageOptPadding)/255.0, flags = cv2.DFT_COMPLEX_OUTPUT)
        filteredSpectra = cv2.mulSpectrums(imageSpectrum, self.WignerKernel, False)
        filteredImage = cv2.idft(filteredSpectra, flags = cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        filteredImage = np.roll(filteredImage, - self.zSize //2, 0)
        filteredImage = np.roll(filteredImage, - self.zSize //2, 1)
        self.filteredImage = 255*np.uint8(filteredImage/np.max(filteredImage.max(axis=0)))
        return self.filteredImage


class imageEnhancing:
    def __init__(self, inputFrame, clipLimit = 1.5, tileGridSize=(4,4)):
        # https://www.mathworks.com/discovery/image-enhancement.html
        self.normalImage = inputFrame
        self.unsharpMask = np.zeros((inputFrame.shape[0], inputFrame.shape[1]), np.uint8)
        self.sharpenedImage = np.zeros((inputFrame.shape[0], inputFrame.shape[1]), np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit, tileGridSize)

    def blurrImage(self, kSize = 5, sigma = 10):
        # http://melvincabatuan.github.io/Image-Filtering/
        self.blurredImage = cv2.GaussianBlur(self.normalImage, (kSize,kSize), sigma, sigma)
        return self

    def unsharpImage(self, sharpening):
        #https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
        self.blurrImage()
        cv2.addWeighted(self.normalImage, 1, self.blurredImage, -1, 0, self.unsharpMask)
        cv2.addWeighted(self.normalImage, 1, self.unsharpMask, sharpening,0,self.sharpenedImage)
        return self

    def LOGImage(self):
        self.blurrImage() 
        self.Laplacian = cv2.Laplacian(self.blurredImage, ddepth=cv2.CV_64F)
        cv2.threshold(self.Laplacian ,0,255.0,cv2.THRESH_TOZERO)
        self.Laplacian  = np.uint8(self.Laplacian) # convert to uint8
        return self

    def unsharpLOGImage(self, sharpening):
        self.LOGImage()
        #cv2.addWeighted(self.normalImage, 1, self.Laplacian, -1, 0, self.unsharpMask)
        cv2.addWeighted(self.normalImage, 1, self.Laplacian, sharpening,0,self.sharpenedImage)
        return self
    
    def equalizeImage(self):
        self.sharpenedImage = cv2.equalizeHist(self.sharpenedImage)
        return self
    
    def applyCLAHE(self):
        self.sharpenedImage = self.clahe.apply(self.sharpenedImage)
        return self
    
    def getProcImage(self):
        return self.sharpenedImage


class imageSegmentation:
    def __init__(self, inputFrame, colorFrame = None, kMrpSize = 7,drawEnabled = False):
        self.pipelineImage = inputFrame
        self.drawEnabled = drawEnabled
        if colorFrame is not None:
            self.colorFlag = True
            self.colorCopy = colorFrame
        else:
            self.colorFlag = False
            self.colorCopy = inputFrame.copy()
        self.morphKernel = np.ones((kMrpSize, kMrpSize), np.uint8)
        pass
        
    def detectEdges(self, sigma = 0, sobelAperture = 20):
        medianValue = np.median(self.pipelineImage)
        
        lowerThreshold = int(max(0, (1.0 - sigma) * medianValue))
        upperThreshold = int(min(255, (1.0 + sigma) * medianValue))
        self.binaryMap = cv2.Canny(self.pipelineImage, lowerThreshold, upperThreshold,sobelAperture)
        #self.attackBlob()
        return self
        
    def createGloblBlobs(self,minTruncLevel = 70, maxTruncLevel = 190):
        self.binaryMap = cv2.threshold(self.pipelineImage,minTruncLevel,\
                        maxTruncLevel, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.attackBlob()
        return self

    def createLocalBlobs(self,truncLevel = 100, blkSize = 13,levelOffset = 5):
        self.binaryMap = cv2.adaptiveThreshold(self.pipelineImage,truncLevel,\
                         cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blkSize,levelOffset)
        self.attackBlob()
        return self
    
    def getMSERcontours(self):
        # https://github.com/opencv/opencv/blob/master/samples/python/mser.py
        localMser = cv2.MSER_create()
        regions = np.array(localMser.detectRegions(self.pipelineImage)[0])
        self.contours = [cv2.convexHull(p.reshape(-1,1,2)) for p in regions]
        return self

    def attackBlob(self):
        self.binaryMap = cv2.erode(self.binaryMap,self.morphKernel) #cv2.morphologyEx(self.binaryMap, cv2.MORPH_OPEN, None)
        return self

    def analyzeShape(self):
        # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
        polySidesData = [] 
        polyHuMData = []
        rectData = []
        sideCounter = []
        centroidData = []
        
        for shape in self.contours:
            hull = cv2.convexHull(shape)
            approx = cv2.approxPolyDP(hull,0.04*cv2.arcLength(hull,True),True)
            sidesApprox = len(approx)

            if sidesApprox == 3:
                polygon = "triangle"
        
            elif sidesApprox == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                polygon = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        
            elif sidesApprox == 5:
                polygon = "pentagon"
        
            else:
                polygon = "circle"
                
            M = cv2.moments(hull)
            # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
            if M["m00"] > 600 and M["m00"] < 40000:
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                
                x,y,w,h = cv2.boundingRect(hull)
                aspect_ratio = float(w)/h
                #https://docs.opencv.org/3.4.3/d1/d32/tutorial_py_contour_properties.html
                if aspect_ratio < 1.3 and aspect_ratio > 0.7:
                
                    polySidesData.append(approx)
                    polyHuMData.append(M)
                    rectData.append([x, y, x + w, y + h])
                    sideCounter.append(sidesApprox)
                    centroidData.append([cX,cY])
                    
                    if self.colorFlag:
                        cv2.rectangle(self.colorCopy,(x,y),(x+w,y+h),(0,255,0),2)
                    else:
                        cv2.rectangle(self.pipelineImage,(x,y),(x+w,y+h),(255,255,255),2)
                        cv2.drawContours(self.pipelineImage, [shape], -1, (0, 255, 0), 2)
                        cv2.putText(self.pipelineImage, polygon, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                
        self.polygonData = [polySidesData, polyHuMData, rectData, sideCounter,centroidData]
        return self

    def getCCLedges(self, connectivity = 8):
        self.connections = np.uint8(cv2.connectedComponents(cv2.Canny(self.binaryMap,0,255),connectivity)[1])
        self.connections[self.connections > 0] = 255 
        return self

    def getImageContours(self):
        self.contours = cv2.findContours(self.binaryMap,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        self.analyzeShape()
        return self
    
    def getCCLContours(self,connectivity = 8):
        self.connections = np.uint8(cv2.connectedComponents(cv2.Canny(self.binaryMap,0,255),connectivity)[1])
        self.connections[self.connections > 0] = 255
        self.contours = cv2.findContours(self.connections,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
        self.analyzeShape()
        return self


class colorSegmentation:
    def __init__(self):
        # Indices: 0 -> Blue_A, 1 -> Blue_B, 2 -> Red, 3 -> Yellow, 4 -> Green, 5 -> Orange
        self.colorMinRanges = [(0,30,20), (175,30,20), (105,60,20),\
                               (60,45,20),(30,30,20), (90,30,20)]
        
        self.colorMaxRanges = [(26,255,255), (180,255,255), (140,255,255),\
                               (105,255,255),(60,255,255), (105,255,255)]
        
        self.morphKernel = np.ones((5,5), np.uint8)
        pass
    
    def maskThresholdColor(self,HSVframe,colorIndex):
        self.singleColorMask = cv2.inRange(HSVframe, self.colorMinRanges[colorIndex], \
                                self.colorMaxRanges[colorIndex])
        return self
    
    def computeAllmasks(self,HSVframe):
        self.colorMasks = [self.maskThresholdColor(HSVframe,colorsIndex).reduceNoise().getSingleMask() for colorsIndex in range(5)]
        return self
    
    def reduceNoise(self):
        self.singleColorMask = cv2.morphologyEx(self.singleColorMask , cv2.MORPH_OPEN, self.morphKernel)
        self.singleColorMask = cv2.dilate(self.singleColorMask,self.morphKernel)
        return self
    
    def getSingleMask(self):
        return self.singleColorMask
    
    def getAllMasks(self):
        return self.colorMasks


class CentroidTracker():
    # https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    # https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
    def __init__(self, maxDisappeared = 4):

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        
    def register(self, centroid, boxData,sides):
        self.objects[self.nextObjectID] = [centroid,boxData,sides]
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        
    def update(self, inputCentroids,inputBoxData,inputSides):
        centroidsCount = len(inputCentroids)
        
        if centroidsCount == 0:
            for objectID in self.disappeared.copy().keys():
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self
            
        if len(self.objects) == 0:
            for i in range(0, centroidsCount):
                self.register(inputCentroids[i],inputBoxData[i],inputSides[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [row[0] for row in list(self.objects.values())]

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = [inputCentroids[col],inputBoxData[col],inputSides[col]]
                self.disappeared[objectID] = 0
                
                usedRows.add(row)
                usedCols.add(col)
                
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputBoxData[col], inputSides[col])

        return self
    
    def drawIndices(self,image):
        boxData = []
        objectLabel = []
        contourSides = []
        for (objectID, contourData) in self.objects.items():
            text = "ID {}".format(objectID)
            centroid = contourData[0]
            boxData.append(contourData[1])
            contourSides.append(contourData[2])
            objectLabel.append(objectID)
            
            try:
                cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            except:
                print(centroid)
            
        return image, objectLabel, boxData, contourSides


class featuresMatcher:
    #https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html
    #https://docs.opencv.org/3.4.3/d1/d89/tutorial_py_orb.html
    #https://docs.opencv.org/3.3.0/db/d27/tutorial_py_table_of_contents_feature2d.html
    #https://www.pyimagesearch.com/2014/04/07/building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/
    def __init__(self, imagesPath, saveDirectories = None):
        self.dataBasePath = imagesPath
        if saveDirectories is None:
            self.saveDir = [imagesPath + '/Red_Entries', imagesPath + '/Yellow_Entries']
        else:
            self.saveDir = saveDirectories
            
        self.pickledDBPath = self.dataBasePath + "/DBFeatures.pck"
        
        #Keypoint descriptors
        self.ORBFeatures = cv2.ORB_create(nfeatures = 50, scaleFactor = 1.5,nlevels = 6, \
                                          edgeThreshold = 51,firstLevel = 0, scoreType = 0,\
                                          patchSize = 51)
        #Texture descriptors
        self.HOGFeatures = cv2.HOGDescriptor(_winSize=(10, 10), _blockSize=(8, 8),\
                                             _blockStride=(1, 1),_cellSize=(8, 8),\
                                             _nbins=9,_derivAperture = 1, _winSigma =-1,\
                                             _histogramNormType = 0,_L2HysThreshold =0.2,\
                                             _gammaCorrection =False,_nlevels =64)
        
        self.ORBMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        self.HOGMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
        
        self.previousLabels = []
        self.dataSet = [[[] for x in range(3)] for y in range(3)]
        pass
    
    def cropImage(self,image,ROIBox):
        image = image[ROIBox[1]:ROIBox[3], ROIBox[0]:ROIBox[2]]
        return image
    
    def describeFeatures(self,image,visualize = False):
        # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
        # https://gurus.pyimagesearch.com/lesson-sample-histogram-of-oriented-gradients-and-car-logo-recognition/
        # https://www.learnopencv.com/histogram-of-oriented-gradients/
        
        image = cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA)
        HOGVector = self.HOGFeatures.compute(image,winStride = (8,8), padding = (8,8))
        
        if visualize:
            keyPoints, ORBDescript = self.ORBFeatures.detectAndCompute(image,None)
            
            HOGImage = feature.hog(image, orientations=3, pixels_per_cell=(8, 8),\
                 cells_per_block=(4, 4), transform_sqrt = False, block_norm="L1",\
                 visualize = True, multichannel=True)[1]
            HOGImage = np.uint8(cv2.normalize(HOGImage, None, 0, 255, cv2.NORM_MINMAX))
            HOGImage = cv2.drawKeypoints(HOGImage, keyPoints, None, color=(0,255,0), flags=0)
            
            return HOGImage, [keyPoints,ORBDescript], HOGVector
        else:
            ORBDescript = self.ORBFeatures.compute(image,None)
            
            return image, ORBDescript, HOGVector
        
    def batchExtractor(self,dataToSave):
        # saving all our feature vectors in pickled file
        with open(self.pickledDBPath, 'wb') as fp: pickle.dump(dataToSave, fp)
        return self
            
    def batchLoader(self):
        with open(self.pickledDBPath,'rb') as fp: data = pickle.load(fp)
        return data
            
    def computeTemplateSet(self):
        if not glob.glob(self.dataBasePath + '/*.pck'):
            for (i,filename) in enumerate(sorted(glob.iglob(self.dataBasePath + '/*.jpg'))):
                
                colorGroupIndex = int(filename.split('_')[2])
                image = cv2.imread(filename)
                image = cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,2]
                if len(image.shape) != 3:
                    imageData, ORBData, HOGData = self.describeFeatures(image, visualize = False)
                else:
                    imageData, ORBData, HOGData = self.describeFeatures(image, visualize = True)
                
                self.dataSet[colorGroupIndex][0].append(image)
                self.dataSet[colorGroupIndex][1].append(ORBData)
                self.dataSet[colorGroupIndex][2].append(HOGData)
                
            self.batchExtractor(self.dataSet)
            
        else:
            self.dataSet = self.batchLoader()
            
        return self
    
    def getTemplateIndex(self,colorGroup,vertexNum):
        if colorGroup == 0:
            if vertexNum == 8:
                index = 1
            elif vertexNum == 3:
                index = 2
            else:
                index = 0
        else:
            index = 0
            
        return index
    
    def computeMatchesVar(self,dataMatches, dataRelRange = 0.6):
        if dataMatches != []:
            dataMatches = sorted(dataMatches, key = lambda x:x.distance)
            distances = [o.distance for o in dataMatches[0:int(len(dataMatches)*dataRelRange)]]
            variance = np.var(distances)
            return variance
        else:
            return[]
        
    def computeHOGMetrics(self, HOGData1, HOGData2, dataRelRange = 1.0):
        dataLimit = int(len(HOGData1)*dataRelRange)
        HOGData1 = HOGData1[0:dataLimit]
        HOGData2 = HOGData2[0:dataLimit]
        directDifference = HOGData1 - HOGData2
        
#        HOGMatches = self.HOGMatcher.match(HOGData1,HOGData2,None)
#        distances = [o.distance for o in HOGMatches]
#        distanceRMS = np.sqrt(np.mean(np.power(distances,2)))
        
        normalizedDIST = np.var(directDifference)/(np.var(HOGData1) + np.var(HOGData2))
        differenceRMS = np.sqrt(np.mean(np.power(directDifference,2)))
#        relativeError = 1 - abs(np.mean((directDifference/HOGData2)))
        
        cosineSimilarity = dist.cosine(HOGData1,HOGData2)
        
        return [cosineSimilarity,differenceRMS,normalizedDIST,np.mean(directDifference)]
        
    def matchFeatures(self,image,objectLabels,ROISets,colorGroup,contourSides,displayFrame):
        # https://docs.opencv.org/3.4.3/dc/dc3/tutorial_py_matcher.html
        
        if ROISets != []:
            initvalidindex = sum(np.in1d(objectLabels,self.previousLabels))
            self.previousLabels = objectLabels
            inputLabelsLen = len(objectLabels)
            
            for validLabel in range(initvalidindex,inputLabelsLen):
                imageROI = self.cropImage(image,ROISets[validLabel])
                
                if imageROI.size != 0:
                    index = self.getTemplateIndex(colorGroup,contourSides)
                    ORBData, HOGData = self.describeFeatures(imageROI)[1:3]
                    
                    ORBMatches = self.ORBMatcher.knnMatch(ORBData[1],self.dataSet[colorGroup][1][index][1],50)
                    ORBVariance = self.computeMatchesVar(ORBMatches)
                    HOGMetrics = self.computeHOGMetrics(HOGData,self.dataSet[colorGroup][2][index])
                    
                    convMatching = cv2.matchTemplate(cv2.resize(imageROI, (80,80), interpolation = cv2.INTER_AREA),self.dataSet[colorGroup][0][index],cv2.TM_CCORR_NORMED)
                    convMatching = cv2.normalize(convMatching, None, 0, 1, cv2.NORM_MINMAX,-1)
                    convMetric = convMatching.mean()
                    
                    print('Index:', index, '-- Group:', colorGroup, '-- ID:', objectLabels[validLabel])
                    print('ORB:', ORBVariance)
                    print('HOG:', HOGMetrics)
                    print('CORR:', convMetric,'\r\n')

                    if convMetric >= 0.55 and HOGMetrics[0] > 0.3 and HOGMetrics[1] < 0.3 and HOGMetrics[2] > 0.8 and HOGMetrics[3] < 0.15:
#                        nameFilename = self.saveDir[colorGroup] + '/ID_' + str(objectLabels[validLabel]) + '_Group_' + str(colorGroup) + '.jpg'
                        nameFilename = os.path.join(self.saveDir[colorGroup],'ID_' + str(objectLabels[validLabel]) + '_Group_' + str(colorGroup) + '.jpg')
                        cv2.imwrite(nameFilename, self.cropImage(displayFrame,ROISets[validLabel]))
                        print('ID', objectLabels[validLabel], 'was saved\r\n')
                    
                else:
                    print('Index:', index, '-- Group:', colorGroup, '-- ID:', objectLabels[validLabel])
                    print('FAILED TO REGISTER\r\n')
                        
            for drawings in range(inputLabelsLen):
                displayFrame = cv2.rectangle(displayFrame.copy(),(ROISets[drawings][0],ROISets[drawings][1]),(ROISets[drawings][2],ROISets[drawings][3]),(0,255,0),3)
            
        return displayFrame
            
    
#%% Camera initialization
widthRes = 640
heightRes= 480
cameraStream = liveStreaming(0, widthRes, heightRes).startThread()
#%% Display Properties
cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', widthRes,heightRes)
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (widthRes,heightRes))
#%% Target Directories
templateDirectory = '/home/pi/dataSet'
saveDirectories = [templateDirectory + '/Red_Entries', templateDirectory + '/Yellow_Entries']
#%% Initialize Features descriptors
colorGroup = [0,1]
colorThreshold = [2,3]
colorSetLenght = len(colorGroup)
spatialMatcher = featuresMatcher(templateDirectory,saveDirectories).computeTemplateSet()
#%% Obtect tracker
blobTracker = CentroidTracker()
#%% Image Enhancers initialization
colorMask = np.zeros((heightRes,widthRes), dtype = "uint8")
#WignerFilter = deconvolutionFilter([heightRes,widthRes])
imageBalancer = colorEnhancing()
colorSegmentator = colorSegmentation();
time.sleep(1.0)
fps = FPS().start()
#%% Frames recording
while True:
    frame = cameraStream.readFrame()
    frame = imageBalancer.balanceImage(frame)
    
    HSVFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) # cv2.COLOR_RGB2GRAY
    #Vframe = WignerFilter.deconvoluteImage(Vframe)
    HSVFrame[:,:,2] = cv2.medianBlur(HSVFrame[:,:,2],3)
    
    preProcessor = imageEnhancing(HSVFrame[:,:,2]).unsharpImage(2).applyCLAHE()
    processedVFrame = preProcessor.getProcImage()
    
    for color in range(colorSetLenght):
        
#        time.sleep(0.025)
        
        colorMask = colorSegmentator.maskThresholdColor(HSVFrame,colorThreshold[color]).reduceNoise().getSingleMask()
        colorLimitedFrame = cv2.bitwise_and(processedVFrame, colorMask)
        
        segmentator = imageSegmentation(colorLimitedFrame,drawEnabled = False).createGloblBlobs().getImageContours()
        imageBorders = segmentator.pipelineImage
        
        pipelineImage, validLabels, ROIs, contourSides = blobTracker.update(segmentator.polygonData[4],\
                                         segmentator.polygonData[2],segmentator.polygonData[3]).drawIndices(imageBorders)
    
        frame = spatialMatcher.matchFeatures(HSVFrame[:,:,2], validLabels,ROIs, colorGroup[color], contourSides, frame)
    
    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # exit order
        break
    fps.update()
    
    #name = 'frame_' + str(8) + '.jpg'
    #cv2.imwrite(name, processedFrame)

#%% Release camera resources
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# When everything done, release the capture
cameraStream.streaming.release()
out.release()
cv2.destroyAllWindows()