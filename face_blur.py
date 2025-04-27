##################################################
########face 객체에만 블러 처리하기#################
##################################################


import torch
import cv2
import time

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='hand_face_best.pt') 

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0: 기본 웹캠
if not cap.isOpened():
    print("Error: 웹캠을 열 수 없습니다.")
    exit()

# FPS 계산용 변수 초기화
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: 프레임을 읽을 수 없습니다.")
        break

    # YOLOv5 모델로 프레임 예측
    results = model(frame)
    
    # 탐지 결과 정보 가져오기
    detections = results.xyxy[0]  # 탐지 결과 (x1, y1, x2, y2, confidence, class)
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)  # 좌표값 정수형으로 변환
        cls_name = 'Hand' if int(cls) == 0 else 'Face'

        # Face 객체만 블러 처리
        if int(cls) == 1:  # Face (클래스 ID 1)
            face_roi = frame[y1:y2, x1:x2]  # 얼굴 영역
            blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)  # GaussianBlur 적용
            frame[y1:y2, x1:x2] = blurred_face  # 블러 처리된 얼굴 다시 삽입
        else:
            # Hand 객체는 시각화만 처리
            color = (0, 255, 0)  # Hand: green
            label = f'{cls_name} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 사각형 그리기
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FPS 계산 및 표시
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 화면에 표시
    cv2.imshow('YOLOv5 Webcam Detection with Face Blur', frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 웹캠 및 OpenCV 창 닫기
cap.release()
cv2.destroyAllWindows()
