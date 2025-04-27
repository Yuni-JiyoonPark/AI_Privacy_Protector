################################################################
###########3초 후 블러 해제###########
################################################################

import torch
import cv2
import time

# YOLOv5 모델 로드 (얼굴과 손 인식)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='hand_face_best.pt')  # 얼굴 및 손 모델 경로

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0: 기본 웹캠

# FPS 계산용 변수 초기화
prev_time = 0

# 얼굴 바운딩박스와 손 바운딩박스가 겹친 시간을 추적하기 위한 딕셔너리
face_status = {}

# 얼굴과 손이 겹치는 비율 계산 함수
def calculate_intersection_area(box1, box2):
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 겹치는 영역의 좌상단과 우하단 좌표 계산
    x_overlap = max(0, min(x2, x2_2) - max(x1, x1_2))
    y_overlap = max(0, min(y2, y2_2) - max(y1, y1_2))

    # 겹치는 영역의 면적 계산
    overlap_area = x_overlap * y_overlap
    return overlap_area

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: 프레임을 읽을 수 없습니다.")
        break

    # YOLOv5 모델로 프레임 예측
    results = model(frame)
    
    # 탐지 결과 정보 가져오기
    detections = results.xyxy[0].cpu().numpy()  # 탐지 결과 (x1, y1, x2, y2, confidence, class)

    # 현재 프레임의 얼굴 ID를 저장
    current_faces = []

    # 탐지된 객체 처리
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)  # 좌표값 정수형으로 변환
        
        if int(cls) == 1:  # Face (클래스 ID 1)
            # 얼굴 객체만 블러 처리
            face_id = f"face_{x1}_{y1}"  # 얼굴 ID 생성
            current_faces.append(face_id)
            face_box = (x1, y1, x2, y2)

            # 얼굴 상태 추적
            if face_id not in face_status:
                face_status[face_id] = {"start_time": None, "blurred": True, "blur_removed": False}

            # 손 객체와 얼굴 바운딩박스 겹침 확인
            for x1_hand, y1_hand, x2_hand, y2_hand, conf_hand, cls_hand in detections:
                if int(cls_hand) == 0:  # 손 (클래스 ID 0)
                    hand_box = (int(x1_hand), int(y1_hand), int(x2_hand), int(y2_hand))
                    overlap_area = calculate_intersection_area(face_box, hand_box)

                    # 손 바운딩박스의 30% 이상이 얼굴 바운딩박스에 겹칠 경우
                    face_area = (x2 - x1) * (y2 - y1)
                    hand_area = (x2_hand - x1_hand) * (y2_hand - y1_hand)
                    overlap_ratio = overlap_area / hand_area

                    if overlap_ratio >= 0.3:  # 겹침 비율 30% 이상
                        if face_status[face_id]["start_time"] is None:
                            face_status[face_id]["start_time"] = time.time()

                        elapsed_time = time.time() - face_status[face_id]["start_time"]
                        if elapsed_time >= 3:  # 3초 이상 겹쳤을 경우 블러 해제
                            face_status[face_id]["blurred"] = False
                            face_status[face_id]["blur_removed"] = True  # 블러 해제 후 해당 얼굴의 상태 유지

            # 블러 처리 또는 해제
            if face_status[face_id]["blurred"] and not face_status[face_id]["blur_removed"]:
                # 얼굴에 블러 처리
                face_roi = frame[y1:y2, x1:x2]  # 얼굴 영역
                blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)  # GaussianBlur 적용
                frame[y1:y2, x1:x2] = blurred_face  # 블러 처리된 얼굴 다시 삽입
            else:
                # 블러 해제된 얼굴은 다시 블러 처리하지 않음
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 블러 해제된 얼굴 표시

        elif int(cls) == 0:  # Hand (클래스 ID 0)
            # 손 객체 시각화: 경계 상자와 확신도 표시
            color = (0, 255, 0)  # 손 객체는 초록색
            label = f'Hand {conf:.2f}'  # 손 객체 라벨
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 경계 상자 그리기
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 확신도 라벨 표시

    # FPS 계산 및 표시
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 화면에 표시
    cv2.imshow('YOLOv5 Webcam Detection with Face Blur and Hand Detection', frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 웹캠 및 OpenCV 창 닫기
cap.release()
cv2.destroyAllWindows()
