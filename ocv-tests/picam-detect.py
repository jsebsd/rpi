#This code is attributed to Raspberry Pi Foundation under a Creative Commons license. 
#The resources is was accessed from https://www.raspberrypi.org/learning/parent-detector/worksheet/

from gpiozero import MotionSensor
from picamera import PiCamera

cam = PiCamera()
pir = MotionSensor(4)
while True:
    pir.wait_for_motion()
    cam.start_preview()
    pir.wait_for_no_motion()
    cam.stop_preview()
