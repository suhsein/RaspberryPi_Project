from flask import Flask, render_template, request, flash, Response
import cv2
import numpy as np
import sys
import threading
import mediapipe as mp
from werkzeug.utils import secure_filename
import os
import servo as sv

# camera
connstr = 'libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! videoscale ! appsink'
cap = cv2.VideoCapture(connstr, cv2.CAP_GSTREAMER)

# face detection, segmentation
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
threshold = 0.02

# 플라스크 웹
app = Flask(__name__)
app.secret_key = 'some_secret'
app.config['IMG_FOLDER'] = os.path.abspath(os.path.join('project', 'static', 'images'))

# 서보모터 딜레이 때문에 스레드로 처리
servo_thread = threading.Thread(target=sv.servo_control, args=())
servo_thread.start()

def gen_frames(filename=''):
	with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
		mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation :
		while True:
			succes, img = cap.read()
			if succes == False:
				print('camera read Failed')
				sys.exit(0)			

			# 웹캠의 프레임 처리. face detection과 segmentation
			# opencv는 BGR순으로 저장하므로 RGB로 변환 후 작업
			# 성능 향상을 위해 img.flags.writeable을 False로 설정한 후 작업(default=True)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img.flags.writeable = False
			face_result = face_detection.process(img)    
			result = selfie_segmentation.process(img)
			img.flags.writeable = True
			
			# 얼굴인식, key point 중 코 부위를 서보모터 자동 제어 함수에 parameter로 전달 
			if face_result.detections:			
				for face in face_result.detections:
					nose = mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.NOSE_TIP)
					sv.servo_facedetect(nose)
					
			# 이미지 출력을 위해 다시 BGR로 변환
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			# 배경 threshold를 0.02로 설정
			condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > threshold

			# 배경 이미지를 업로드 한 경우
			if filename != '':
				path = os.path.join(app.config['IMG_FOLDER'], filename)
				bg_img = cv2.imread(path)	
				bg_img = cv2.resize(bg_img, (640,480))
				img = np.where(condition, img, bg_img)

			# 스위치 눌렸다면 이미지 캡쳐 함수 실행		
			sw_val = sv.readadc(sv.sw_channel)
			if sw_val < 100:
				sv.img_capture(app.config['IMG_FOLDER'], sw_val, img)
			
			# 이미지 인코딩 / 프레임 쌓기
			ref, buffer = cv2.imencode('.jpg', img)   
			img = buffer.tobytes()
			yield (b'--frame\r\n'
					b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  

# 인덱스 페이지
@app.route('/')
def home():
	print(app.config['IMG_FOLDER'])
	return render_template('index.html')

# 배경 이미지 업로드 처리
@app.route('/uploader', methods=['POST'])
def file_upload():
	if request.method == 'POST':
		file = request.files['chooseFile']
		path = os.path.join(app.config['IMG_FOLDER'], secure_filename(file.filename))
		file.save(path)
	return render_template('index.html', filename=file.filename)

# 배경 이미지 X
@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame') 

# 배경 이미지 O
# 플라스크의 동적 페이지 라우팅 방식을 통해 배경 이미지에 따라서 프레임을 변경
@app.route('/video_feed/<filename>')
def video_feed2(filename):
	return Response(gen_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame') 
	
if __name__ == "__main__":
	app.run(host="172.30.1.37", port="8080")
