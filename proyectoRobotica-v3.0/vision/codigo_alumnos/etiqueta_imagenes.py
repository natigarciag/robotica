import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import select_pixels as sel
import numpy as np


capture = cv2.VideoCapture('./video.mp4')

numberOfImages = 10

hsImages = np.memmap('hsImages.driver', dtype='uint8', mode='w+', shape=(numberOfImages, 240, 320, 2))
markedImages = np.memmap('markedImages.driver', dtype='uint8', mode='w+', shape=(numberOfImages, 240, 320, 3))

for i in range(numberOfImages):
    print(i)

    # cImg = 0
    key = 0
    while (key != 27):
        ret, im = capture.read()

        cv2.imshow('Captura', im)

        # if (key == ord('g')):
        #     # imsave('kk{0}.png'.format(cImg), im[:,:,(2,1,0)])
        #     cImg += 1
        
        key = cv2.waitKey(35)

    cv2.destroyWindow('Captura')
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    imNp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    markImg = sel.select_fg_bg(imNp, radio=5)
    
    # imsave('kk10.png', markImg[:,:,(2,1,0)])

    # plt.imshow(markImg)
    # plt.show()


    hsImages[i] = cv2.cvtColor(imNp, cv2.COLOR_RGB2HSV)[:,:,(0,1)]
    markedImages[i] = markImg

hsImages.flush()
markedImages.flush()





# # Abres el video / camara con

# capture = cv2.VideoCapture('./video.mp4')

# # Lees las imagenes y las muestras para elegir la(s) de entrenamiento
# # posibles funciones a usar

# # cv2.waitKey()
# # capture.read()
# # cv2.imshow()

# # capture.release()
# # cv2.destroyWindow("Captura")

# while(True):
#     # Capture frame-by-frame
#     ret, frame = capture.read()

#     # Our operations on the frame come here
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Si deseas mostrar la imagen con funciones de matplotlib posiblemente haya que cambiar
# # el formato, con
# # cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)

# # Esta funcion del paquete "select_pixels" pinta los pixeles en la imagen 
# # Puede ser util para el entrenamiento

# markImg = sel.select_fg_bg(imNp)

# # Tambien puedes mostrar imagenes con las funciones de matplotlib
# plt.imshow(markImg)
# plt.show()

# # Si deseas guardar alguna imagen ....

# imsave('lineaMarcada.png',markImg)

