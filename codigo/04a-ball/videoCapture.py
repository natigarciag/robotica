import numpy as np
import cv2
import os
import cameraCapture

cap = cameraCapture.capture

if not os.path.exists('./capturedVideos'):
    os.makedirs('./capturedVideos')
out = cv2.VideoWriter('./capturedVideos/bolaTenisAlpera_2019_05_23_17_17.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (320, 240))

# print cv2.CAP_PROP_EXPOSURE

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
