# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044

import numpy as np
import cv2
from matplotlib import pyplot as plt
import config
import math
import os
import time
import setup

showRawImage = True
rawImageShrinkFactor = 1
showSegmentedImage = True
segmentedImageShrinkFactor = 1
showArrowSegmentation = True
arrowSegmentationShrinkFactor = 1
showLineSegmentation = False
lineSegmentationShrinkFactor = 1

times = []
try:
    while (setup.capture.isOpened()):
        beg = time.time()
        
        ret, im = setup.capture.read()
        if showRawImage:
            cv2.imshow('raw image', im[0::rawImageShrinkFactor,0::rawImageShrinkFactor,:])

        # prepare segmentation
        imHSV = im[((setup.originalImageHeight - setup.imageHeight)*setup.shrinkFactor)::setup.shrinkFactor, 0::setup.shrinkFactor, :]
        imHSV = cv2.cvtColor(imHSV, cv2.COLOR_BGR2HSV)
        imHS = imHSV[:,:,(0,1)]
        
        # perform segmentation
        def predictRow(i):
            setup.segImg[i] = setup.segmenter.predict(imHS[i])
        [predictRow(i) for i in range(imHS.shape[0])]

        # separate symbols from lines
        arrow = np.zeros(setup.segImg.shape, dtype='uint8')
        line = np.zeros(setup.segImg.shape, dtype='uint8')
        # 0 - line
        # 1 - floor
        # 2 - symbols
        # 3 - nothing - not used
        line[setup.segImg == 0] = 1
        arrow[setup.segImg == 2] = 1
        arrow = cv2.erode(arrow, None, dst=arrow, iterations=1)
        
        # if showSegmentedImage:
        #     cv2.imshow('segmented image', cv2.cvtColor(paleta[setup.segImg], cv2.COLOR_RGB2BGR))
        showImg = np.copy(setup.segImg)
        showImg[showImg == 2] = 1
        showImg[arrow == 1] = 2
        if setup.showSegmentedImage:
            cv2.imshow('segmented treated image', cv2.cvtColor(setup.paleta[showImg][0::segmentedImageShrinkFactor,0::segmentedImageShrinkFactor,:], cv2.COLOR_RGB2BGR))

        if showArrowSegmentation:
            cv2.imshow('arrows', (arrow*255)[0::arrowSegmentationShrinkFactor,0::arrowSegmentationShrinkFactor])
        if showLineSegmentation:
            cv2.imshow('line', (line*255)[0::setup.lineSegmentationShrinkFactor,0::lineSegmentationShrinkFactor])

        touchesEdges = setup.touchingEdges(arrow, 1)

        if(np.sum(arrow) > int(200 / (setup.shrinkFactor*setup.shrinkFactor))):
            if (not touchesEdges):
                moments = cv2.HuMoments(cv2.moments(arrow)).flatten()
                # print(moments)
                predictedShape = setup.symbolClassifier.predict(np.array([moments]))
                # print(predictedShape, namesOfTheShapes[int(predictedShape[0])])
                # print(predictedShape)
                print(setup.namesOfTheShapes[int(predictedShape[0])])
            # else:
            #     print('touches edges')
        # else:
        #     print('nothing')

        times.append(time.time() - beg)

        cv2.waitKey(1)
    
except TypeError as a:
    print(a)
finally:
    # print(times)
    print(np.mean(np.array(times)), 'was the time per frame')