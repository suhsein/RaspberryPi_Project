import RPi.GPIO as GPIO
import pigpio
from datetime import datetime
import time
import spidev
import os
import cv2

# GPIO SET
GPIO.setwarnings(False); GPIO.setmode(GPIO.BCM)	

# servo motor
pi = pigpio.pi(); SERVO_PIN = 18;			# 성능 향상을 위해 pigpio 사용
pw_MIN = 1000; pw_MAX = 2000; delay = 0.2
pw = (pw_MIN+pw_MAX)/2; pi.set_servo_pulsewidth(SERVO_PIN, pw)

# joy stick
sw_channel = 0; vry_channel = 1; vrx_channel = 2
spi = spidev.SpiDev(); spi.open(0,0); spi.max_speed_hz = 1000000 

# SPI 채널 값 읽어오기
def readadc(adcnum):
	if adcnum > 7 or adcnum < 0:
		return -1
	r= spi.xfer2([1,(8 + adcnum) << 4, 0])
	data = ((r[1] & 3) << 8) + r[2]
	return data

# 서보모터 수동제어
def servo_control():
	global pw
	while True:
		vrx_pos = readadc(vrx_channel)
		if vrx_pos < 300:
			pw = max(pw-10, pw_MIN)
		elif vrx_pos > 700:
			pw = min(pw+10, pw_MAX)
		pi.set_servo_pulsewidth(SERVO_PIN, pw)
		time.sleep(delay)

# 서보모터 스위치로 이미지 캡쳐
# 시간으로 파일 이름
def img_capture(img_folder, img):
    now = datetime.now()
    filename = now.strftime('%Y-%m-%d_%H-%M-%S') + '.jpg'
    path = os.path.abspath(os.path.join(img_folder, filename))
    cv2.imwrite(path, img)
    print('img capture')

# face detect로 서보모터 자동제어
def servo_facedetect(nose):
	global pw
		
	if nose.x < 0.4:
		pw = min(pw+10, pw_MAX)
	elif nose.x > 0.6:
		pw = max(pw-10, pw_MIN)
	pi.set_servo_pulsewidth(SERVO_PIN, pw)

