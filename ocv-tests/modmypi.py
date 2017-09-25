import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
PIR = 7
GPIO.setup(PIR,GPIO.IN)

def MOTION(PIR):
	print "Motion Detected!"

print "PIR Module test (Ctrl+C to exit)"
time.sleep(2)
print "Ready"

try:
	GPIO.add_event_detect(7,GPIO.RISING,callback=MOTION)
	while 1:
		time.sleep(100)
except KeyboardInterrupt:
	print " Quit"
	GPIO.cleanup()
