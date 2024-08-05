import cv2 as cv
import numpy as np
videoCapture = cv.VideoCapture('traffic.mp4')
object = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    ret, frame = videoCapture.read()
    height, width, _ = frame.shape
    blur = cv.GaussianBlur(frame, (31, 31), 0)
    mask = object.apply(blur)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
            #cv.drawContours(frame, [cnt], -1, (0, 255, 0), 1)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    if not ret: break
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(grayFrame, 80, 255, cv.THRESH_BINARY)
    cv.imshow("blur", blur)
    cv.imshow("box", frame)
    if cv.waitKey(1) & 0xff == ord('q'): break
videoCapture.release()
cv.destroyAllWindows()