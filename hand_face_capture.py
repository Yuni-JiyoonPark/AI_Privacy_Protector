###############################################################
#############손 감지(2초) 후 이미지 저장 (countdown)#############
###############################################################


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

# 이미지 저장 폴더 설정 (폴더가 없으면 생성)
output_dir = "captured_images"

# 캡처된 이미지 저장을 위한 변수 설정
capture_triggered = False
capture_time = 0
capture_index = 1  # 저장할 이미지 번호 초기화
hand_detection_time = 0  # 손이 감지된 시간 추적용 변수

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: 프레임을 읽을 수 없습니다.")
        break

    # YOLOv5 모델로 프레임 예측
    results = model(frame)

    # 탐지 결과에서 손 객체 감지 확인
    detected_objects = results.pandas().xywh[0]  # 'xywh' 컬럼을 가진 데이터프레임
    hand_detected = False
    for _, obj in detected_objects.iterrows():
        if obj['name'] == 'hand':  # 객체 이름이 'hand'인 경우
            hand_detected = True
            break

    # 손 객체가 2초 이상 감지되면 카운트다운 시작
    if hand_detected:
        if hand_detection_time == 0:
            hand_detection_time = time.time()  # 손이 처음 감지된 시간 기록
        elif time.time() - hand_detection_time >= 2:  # 손이 2초 이상 감지된 경우
            if not capture_triggered:
                capture_triggered = True
                capture_time = time.time()
                print("손 객체가 2초 이상 감지됨. 3초 카운트다운 시작!")

    else:
        hand_detection_time = 0  # 손이 감지되지 않으면 시간 초기화

    if capture_triggered:
        # 3초 카운트다운
        countdown_time = 3 - int(time.time() - capture_time)
        if countdown_time > 0:
            cv2.putText(frame, f"Capture after {countdown_time}sec", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # 순차적인 파일 이름으로 저장
            capture_filename = f"captured_images/captured_image_{capture_index}.jpg"  # 경로를 직접 문자열로 지정
            cv2.imwrite(capture_filename, frame)
            print(f"이미지 저장됨: {capture_filename}")

            # 파일 번호 증가
            capture_index += 1
            capture_triggered = False  # 카운트다운이 끝났으므로 리셋

    # 탐지된 결과를 이미지에 시각화
    annotated_frame = results.render()[0]  # YOLOv5의 렌더링된 결과 가져오기

    # 화면에 표시
    cv2.imshow('YOLOv5 Webcam Detection', annotated_frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 웹캠 및 OpenCV 창 닫기
cap.release()
cv2.destroyAllWindows()