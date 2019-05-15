import cv2
import numpy as np
import math

def fetchImageAndSegment(setup):
    ret, im = setup.capture.read()
    if setup.showRawImage:
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

    imageOnPaleta = setup.paleta[showImg]

    if setup.showArrowSegmentation:
        cv2.imshow('arrows', (arrow*255)[0::setup.arrowSegmentationShrinkFactor,0::setup.arrowSegmentationShrinkFactor])
    if setup.showLineSegmentation:
        cv2.imshow('line', (line*255)[0::setup.lineSegmentationShrinkFactor,0::setup.lineSegmentationShrinkFactor])

    return arrow, line, imageOnPaleta
