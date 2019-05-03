# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044
 
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_SATURATION, 150)

imageSeq = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame[::4,::4,:])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # store image
        cv2.imwrite('./images/' + str(imageSeq) + '.png',frame)
        imageSeq += 1
        print imageSeq

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
