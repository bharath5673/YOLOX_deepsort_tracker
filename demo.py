from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os


cap = cv2.VideoCapture(0)
# tracker = Tracker(model='yolox-s', ckpt='YOLOX/weights/yolox_s.pth',filter_class=['truck','person','car'])
tracker = Tracker(model='yolox-s', ckpt='YOLOX/weights/yolox_s.pth',filter_class=['person'])
# tracker = Tracker()
while True:
    _, im = cap.read()
    if im is None:
        break
    im = imutils.resize(im, height=640)
    image,_ = tracker.update(im)
   

    cv2.imshow('demo', image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows() 
