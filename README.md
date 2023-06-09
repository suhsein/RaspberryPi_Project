# 인물 트래킹 카메라와 웹 포토부스

## 📋Table of Contents
1. [🫲프로젝트 소개🫱](#프로젝트-소개)
2. [⏳개발기간](#개발기간)
3. [📚기술스택](#기술스택)
4. [🤖사용부품](#사용부품)
5. [🔎기능](#기능)
6. [🌠TroubleShooting](#TroubleShooting)
7. [💡Takeaway](#Takeaway)

*******

## 🫲프로젝트 소개🫱
*2023-1 임베디드 시스템 과목의 라즈베리파이 프로젝트*<br/><br/>
__인물 트래킹 카메라 & 웹 포토부스__ 입니다.<br/>
mediapipe의 face detection, fragmentation을 이용해 구현하였습니다.<br/>
**하단의 이미지를 클릭하면 코드 설명 및 시연 동영상을 볼 수 있습니다. <br/>** <br/>
[![Video Label](http://img.youtube.com/vi/m6-DWYGrjWw/0.jpg)](https://youtu.be/m6-DWYGrjWw)
<br/><br/>

## ⏳개발기간
2023.06.01~2023.06.15(2주)
<br/><br/>

## 📚기술스택
<img src="https://img.shields.io/badge/RaspberryPi-A22846?style=flat&logo=RaspberryPi&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white"/>
<img src="https://img.shields.io/badge/MediaPipe-4285F4?style=flat&logo=Google&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-000000?style=flat&logo=Flask&logoColor=white"/>
<img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=HTML5&logoColor=white"/>
<img src="https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=CSS3&logoColor=white"/>
<img src="https://img.shields.io/badge/VSCode-007ACC?style=flat&logo=VisualStudioCode&logoColor=white"/>
<img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=GitHub&logoColor=white"/>
<br/><br/>

## 🤖사용부품
* 라즈베리파이4
* 라즈베리파이 카메라 모듈
* 조이스틱
* 서보모터
* 컵홀더와 편지지로 카메라 거치대 제작
* 그 외 점퍼선, 충전기, 브레드보드 등
<br/>

## 🔎기능

### 기능 1. 서보모터를 이용한 카메라 움직임<br/>
서보혼에 카메라 거치대를 달아서 카메라가 수평방향으로 움직일 수 있도록 했습니다.<br/>
서보모터 조작은 조이스틱을 이용한 수동제어와 face detection을 이용한 자동제어가 가능합니다.<br/>
인물이 카메라 밖에 있을 때는 조이스틱을 이용해서 수동제어를 할 수가 있습니다.<br/>
만약 인물이 카메라 안에 들어오는 경우에는 인물을 프레임의 가운데에 맞추기 위해서 서보모터가 자동제어 됩니다.<br/>

### 기능 2. 웹 서버에서 라이브 스트리밍<br/>
웹캡을 통해서 촬영되는 영상은 플라스크를 통해 구동된 서버에 라이브 스트리밍 됩니다.<br/>
사용자는 해당 주소에 접속하여 실시간으로 스트리밍 되는 영상을 확인할 수 있습니다.<br/>

### 기능 3. 카메라 캡쳐<br/>
조이스틱의 스위치 클릭을 통해서 카메라 캡쳐가 가능합니다.<br/>
캡쳐된 이미지는 static/images 폴더에 저장되며, 파일명은 현재 날짜와 시간입니다.<br/>

### 기능 4. 배경 바꾸기<br/>
segmentation을 이용해 인물과 배경을 분리해낼 수 있습니다.<br/>
choosefile 버튼을 클릭하여 사용자 컴퓨터로부터 원하는 배경 이미지를 선택한 후, submit 버튼을 클릭하면, 해당 이미지로 배경이 바뀝니다.<br/>
배경 이미지 또한 static/images 폴더에 저장됩니다.<br/>
배경을 바꾼 상태로 카메라 캡쳐를 할 수 있습니다.<br/><br/>

## 🌠TroubleShooting

### 1.	카메라 거치대 만들기
인터넷에서 카메라 서보모터를 사용한 카메라 거치대를 팔고있긴 했지만, 직접 만들어보고 싶었기에 재료와 부착 방법 등을 생각하는데 어려움이 있었다.

### 2.	카메라 모듈 사용
라즈베리파이 os bullseye 업데이트로 인해 카메라 사용 방법에 많은 변화가 있어서 어려움이 있었다.
opencv를 설치할 때도 os가 32bit가 아닌 64bit일 때만 설치가 가능해서 os를 처음부터 다시 설치해야 했다.
opencv video capture 사용 시 변수로 0이 아닌 gstreamer를 사용해 주어야 했다.

### 3.	서보모터 jitter-shaking 현상
서보모터만 조작하도록 했을 때는 문제없이 작동했는데 웹캠과 함께 서보모터를 제어하니 서보모터가 달달 떨리는 현상이 일어났다. pigpio 모듈을 설치해서 작동을 시켰더니 해당 현상이 일어나지 않고 잘 동작하였다.

### 4.	opencv face detection이 너무 느림
처음에는 opencv의 face detection을 사용했는데 너무 느려서 스트리밍을 할 때 mediapipe의 계속딜레이가 생겼다. mediapipe 모듈의 face detection을 사용해서 문제를 해결할 수 있었다.

### 5.	서보모터 딜레이
서보모터 조작 시 움직이는 시간을 위해 딜레이를 주었는데 카메라도 같이 딜레이가 걸렸다. 서보모터 조작을 위한 스레드를 따로 만들어서 해결했다.


### 6.	flask 사용
파이썬 파일을 웹에서 구동하고 싶었는데 파이썬 웹 프레임워크를 사용해본 적이 없었다. 플라스크가 장고보다는 가벼운 프레임워크이므로 플라스크를 선택했고, 짧게 공부하여 웹 서버를 구동할 수 있었다. 
post 방식으로 이미지를 받아와서 동적 페이지 라우팅을 하는 부분이 가장 어려웠다.<br/><br/>


## 💡Takeaway
<p>라즈베리 파이 프로젝트는 임베디드 시스템 과목의 최종 프로젝트였다. 수강생 전원에게 라즈베리 파이가 배포되었고, 같은 라즈베리 파이와 부품들이 주어졌을 때 각자 다른 결과물을 만들어내는 것이 신기하고 재밌었다.
마치 모두에게 같은 삶이 주어지지만 어떤 사람이 되느냐는 스스로에게 달려있는 것과 같이 라즈베리 파이에 숨을 불어넣는 과정이라고 생각했다.</p>
<p>인물 트래킹 카메라는 시중에 존재하는 제품이긴 하지만 요즘 같은 디지털 미디어의 시대, 나만의 개성을 뽐내고 싶은 시대에 나만을 따라다니는 카메라, 나만을 위한 포토부스를 만들어보는 것이 프로젝트의 처음 기획 의도였다.
그 과정에서 파이썬의 웹 애플리케이션을 만들기 위한 플라스크라는 프레임 워크도 알게 되었고, 동적 라우팅 방식에 대해 더 공부할 수 있었다. 소프트웨어 뿐만 아니라 회로를 직접 연결하고 거치대를 만드는 하드웨어 구성 과정으로 손맛을 느꼈고, 회로에 대해 이해할 수 있었다.</p>
<p>개발 초기 opencv와 mediapipe 설치를 하는 데에 오류와 재설치를 반복하며 몇 시간 내에 끝날 줄 알았던 일들이 며칠이 걸리는 기적을 체험하였다.
이번 경험을 통해 앞으로 개발 초기에 어떤 모듈을 설치하게 되더라도 이번 프로젝트보다는 더 빨리 설치할 수 있을 것이라는 확신과 배움을 얻었다.</p>
다음 프로젝트에서는 예외처리 부분에 대해서 더 고려하고 싶고, 데이터베이스를 사용하는 프로젝트를 진행해보고 싶다.

