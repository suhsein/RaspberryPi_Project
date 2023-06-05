from flask import Flask, render_template, redirect, url_for, request, flash, Response
from fileinput import filename
import RPi.GPIO as GPIO
import time
import spidev
import cv2
import numpy as np
import sys
import threading
import pigpio
import mediapipe as mp
from werkzeug.utils import secure_filename
import os

# camera
connstr = 'libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! videoscale ! clockoverlay time-format="%D %H:%M:%S" ! appsink'
cap = cv2.VideoCapture(connstr, cv2.CAP_GSTREAMER)

# face detection, segmentation
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

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
	global pw, img
	while True:
		vrx_pos = readadc(vrx_channel); sw_val = readadc(sw_channel)
		if vrx_pos < 300:
			pw = max(pw-10, pw_MIN)
		elif vrx_pos > 700:
			pw = min(pw+10, pw_MAX)
		pi.set_servo_pulsewidth(SERVO_PIN, pw)
		if sw_val < 100:
			cv2.imwrite('img.jpg', img)
			print('img capture')
		time.sleep(delay)

# face detect로 서보모터 자동제어
def servo_facedetect(nose):
	global pw
		
	if nose.x < 0.4:
		pw = min(pw+10, pw_MAX)
	elif nose.x > 0.6:
		pw = max(pw-10, pw_MIN)
	pi.set_servo_pulsewidth(SERVO_PIN, pw)


servo_thread = threading.Thread(target=servo_control, args=())
servo_thread.start()

# 플라스크 웹
app = Flask(__name__)
app.secret_key = 'some_secret'
app.config['IMG_FOLDER'] = os.path.abspath(os.path.join('project', 'static', 'images'))


def gen_frames(filename=''):
	with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
		mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation :
		while True:
			succes, img = cap.read()
			if succes == False:
				print('camera read Failed')
				sys.exit(0)			

			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img.flags.writeable = False
			face_result = face_detection.process(img)    
			result = selfie_segmentation.process(img)
			img.flags.writeable = True
			
			if face_result.detections:			
				for face in face_result.detections:
					nose = mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.NOSE_TIP)
					servo_facedetect(nose)
					
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.02

			if filename != '':
				path = os.path.join(app.config['IMG_FOLDER'], filename)
				bg_img = cv2.imread(path)	
				bg_img = cv2.resize(bg_img, (640,480))
				img = np.where(condition, img, bg_img)
			
			ref, buffer = cv2.imencode('.jpg', img)   
			img = buffer.tobytes()
			yield (b'--frame\r\n'
					b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  
			
@app.route('/')
def home():
	print(app.config['IMG_FOLDER'])
	return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def file_upload():
	if request.method == 'POST':
		file = request.files['chooseFile']
		path = os.path.join(app.config['IMG_FOLDER'], secure_filename(file.filename))
		file.save(path)
	return render_template('index.html', filename=file.filename)

@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame') 

@app.route('/video_feed/<filename>')
def video_feed2(filename):
	return Response(gen_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame') 
	
if __name__ == "__main__":
	app.run(host="172.30.1.37", port="8080")
