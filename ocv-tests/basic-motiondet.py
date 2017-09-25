#This code is attributed to Raspberry Pi Foundation under a Creative Commons license. 
#The resources is was accessed from https://www.raspberrypi.org/learning/parent-detector/worksheet/

from gpiozero import MotionSensor
import time

pir = MotionSensor(4)
 
print "Press Ctrl + C to quit"
time.sleep(10)
try:
	while True:
		if pir.motion_detected:
			print "Motion detected! (Ctrl + C to quit)"
except KeyboardInterrupt:	#exits when CTL+C is pressed
	print "  Exit"

