#######################################
######얼굴 감지&블러 + 손 감지 모델######
#######################################

import cv2
import torch
import numpy as np

# YOLOv5 모델 로드 (손바닥 감지용)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Admin/yolov5/runs/train/exp3/weights/best.pt')  # 손바닥 모델 경로

# OpenCV Haar Cascade 얼굴 감지 모델 로드
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Haar Cascade XML 파일 경로

# 웹캠 실행
cap = cv2.VideoCapture(0)  # 0번 카메라 (웹캠)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5를 사용하여 손바닥 감지
    results = model(frame)
    hand_detections = results.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표 (x1, y1, x2, y2)

    # OpenCV로 얼굴 감지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 얼굴 감지를 위한 그레이스케일 변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 손바닥 감지 결과 표시
    for x1, y1, x2, y2, conf, cls in hand_detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 손바닥 바운딩 박스
        cv2.putText(frame, f'Hand {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 얼굴 감지 결과에 블러 적용
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_region = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)  # 블러 적용
        frame[y:y+h, x:x+w] = blurred_face  # 블러 처리된 얼굴로 교체

    # 결과 출력
    cv2.imshow('Hand Detection and Face Blur', frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 웹캠 종료
cap.release()
cv2.destroyAllWindows()
