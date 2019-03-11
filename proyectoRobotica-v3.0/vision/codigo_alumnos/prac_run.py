####################################################
# Esqueleto de programa para ejecutar el algoritmo de segmentacion.
# Este programa primero entrena el clasificador con los datos de
#  entrenamiento y luego segmenta el video (este entrenamiento podria
#  hacerse en "prac_ent.py" y aqui recuperar los parametros del clasificador
###################################################


import cv2
# from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
import config
import datasetGenerator
import clasificadorEuc

import time



clf = clasificadorEuc.Clasificador(datasetGenerator.shapeD)
clf.train()

# Inicio la captura de imagenes
capture = cv2.VideoCapture('./video.mp4')

# Ahora clasifico el video
while (True):
    ret, im = capture.read()

    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    cv2.imshow('Real', im)

    # segmentation
    imHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    imHS = imHSV[:,:,(0,1)]

    paleta = np.array([[0,0,0],[0,0,255],[0,255,0],[255,0,0]],dtype='uint8')  

    # print(imHS[1,2])

    # segImg = np.array([[paleta[clf.predict(imHS[i,j])] for j in range(imHS.shape[1])] for i in range(imHS.shape[0])], dtype='uint8')
    segImg = np.array([paleta[clf.predict(imHS[i])] for i in range(imHS.shape[0])], dtype='uint8')
    # print segImg

    # print(segImg[1,2])
    # print(segImg.shape)
    
    # cv2.imshow('Segmentacion', segImg)
    # print segImg.shape
    cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))
    # cv2.imshow('Segmentacion', segImg)

    # time.sleep(1)

    # key = cv2.waitKey(35)
    cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)

cv2.destroyWindow('Captura')

cv2.waitKey(1)
cv2.waitKey(1)

    # imNp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # markImg = sel.select_fg_bg(imNp, radio=4)
    
    # # imsave('kk10.png', markImg[:,:,(2,1,0)])

    # # plt.imshow(markImg)
    # # plt.show()


    # hsImages[i] = cv2.cvtColor(imNp, cv2.COLOR_RGB2HSV)[:,:,(0,1)]
    # markedImages[i] = markImg

    # # voy a segmentar solo una de cada 25 imagenes y la muestro
    # # ........
    # cv2.imshow("Imagen",img)

    # ret, im = capture.read()

    #     cv2.imshow('Captura', im)

    # # La pongo en formato numpy

    # # Segmento la imagen.
    # # Compute rgb normalization
    # imrgbn=np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]
    
    # labelsEu=segmEuc.segmenta(imNp)
    # labelsMa=segmMano.segmenta(imNp)


    # # Vuelvo a pintar la imagen
    # # genero la paleta de colores
    # paleta = np.array([[0,0,0],[0,0,255],[0,255,0],[255,0,0]],dtype=np.uint8)
    # # ahora pinto la imagen
    # cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[labelsEu],cv2.COLOR_RGB2BGR))
    # cv2.imshow("Segmentacion Mano",cv2.cvtColor(paleta[labelsMa],cv2.COLOR_RGB2BGR))

    # # Para pintar texto en una imagen
    # cv2.putText(imDraw,'Lineas: {0}'.format(len(convDefsLarge)),(15,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0))
    # # Para pintar un circulo en el centro de la imagen
    # cv2.circle(imDraw, (imDraw.shape[1]/2,imDraw.shape[0]/2), 2, (0,255,0), -1)

    # # Guardo esta imagen para luego con todas ellas generar un video
    # cv2.imwrite

