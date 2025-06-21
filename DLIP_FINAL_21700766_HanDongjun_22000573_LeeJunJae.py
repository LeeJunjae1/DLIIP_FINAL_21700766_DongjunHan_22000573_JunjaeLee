###################################################
# DLIP FINAL 
# @author 21700766 Dongjun Han, 22000573 Junjae Lee
# @Mod      2025 - 06 - 20
####################################################

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import serial
import math
import pygame

last_event_sent = ""
last_possession_sent = ""

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 시리얼 포트 연결, ESP-32와 연동

# COM3의 경우 변동이 필요. 사용자의 설정에 따라 변경 필요
try:
    ser = serial.Serial('COM3', 115200)
    print("시리얼 포트 연결 성공")
except serial.SerialException as e:
    print("시리얼 포트 연결 실패:", e)
    ser = None

    

video_path = "./soccer/new3_red_blue.mp4" #비디오 경로
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8x.pt") #기존에 yolo 모델

# 프레임 너비와 높이
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
fps = 30

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

output_path="./output_video/NEW_result_red_blue.mp4" #저장 경로

#저장 파일 형식
out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps,
                      (frame_width, frame_height))

player_last_positions = {}
player_distances = defaultdict(float)
pixel_to_meter = 0.02 # 픽셀당 m 변환
player_colors = {}
frame_count = 0

ball_positions = []  # 공 위치 기록 리스트 (최근 30프레임 유지)
ball_possession_count = {'Red Team': 1, 'Blue Team': 1}  # 50:50 시작
ball_owner_distance_thresh = 100

# 필드 라인따기
def get_border_mask(frame):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    #ROI 선택 진행
    outer_polygon = np.array([[750, 40], [480, 995], [1760, 995], [1350, 40]])
    cv2.fillPoly(mask, [outer_polygon], 255)
    inner_polygon = np.array([[780, 60], [550, 955], [1680, 955], [1320, 60]])
    cv2.fillPoly(mask, [inner_polygon], 0)
    return mask

def line_equation(x1, y1, x2, y2):
    # y = mx + b 형태의 직선 방정식 반환 (m, b)
    if x2 - x1 == 0:  # 수직선 처리
        return None, x1
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def intersection(line1, line2):
    # 두 직선의 교점 계산
    m1, b1 = line1
    m2, b2 = line2
    
    # 수직선 처리: m=None 이면 x=b 가 직선
    if m1 is None and m2 is None:
        return None  # 평행 수직선 (교점 없음)
    if m1 is None:  # line1 수직선
        x = b1
        y = m2 * x + b2
        return int(x), int(y)
    if m2 is None:  # line2 수직선
        x = b2
        y = m1 * x + b1
        return int(x), int(y)
    
    if m1 == m2:
        return None  # 평행선
    
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return int(x), int(y)

detected_lines = [] # 라인 검출
first_frame = True # 첫 프레임에서만 적용하기 위해
trapezoid_pts = np.array([[750, 40], [480, 995], [1760, 995], [1350, 40]]) # 관심없는 영역을 어둡게 하기 위한 좌표

goal_flag = False # 골여부 판단
goal_frame_counter = 0 #골 문자 출력을 위한 변수
goal_display_frames = 90  # 골 이미지 표시 프레임 수

# 변수로 노란선 경계 저장
yellow_line = None  # (x_left, x_right, y_left, y_right)

throwin_line_x = None  # 기준선 x값 저장
throwin_flag = False #스로잉 flag
goal_line_flag = False #골 라인 out flag
red_team_goal_flag =False #red 팀 골 판정
blue_team_goal_flag =False #blue 팀 골 판정

person_near_ball = False # 공과 가까이 있는지에 대한 여부
throwin_reset_pending = False  # 사람이 가까웠던 이력이 있는지 추적
goal_frame = None #골을 넣은 순간의 프레임 번호
highlight_margin = 150 #골 넣는 순간을 기준으로 저장할 앞, 뒤 프레임 수
highlight_frames = []
mp3_path="./soccer/0001.mp3" #음악 저장 경로
goal_sound_played = False
def play_goal_sound():
    playsound(mp3_path)
# 음악 세팅 초기화
pygame.mixer.init()
goal_sound = pygame.mixer.Sound(mp3_path)




while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()    
    #frame_line = frame.copy()
    border_mask = get_border_mask(frame) #경계선 마스킹 진행
    masked_frame = cv2.bitwise_and(frame, frame, mask=border_mask) # 프레임에 마스킹 적용

    # frame_copy = frame.copy()  


    # 회색 바탕, 관심 없는 영역 어둡게 하기 위해
    trapezoid_pts = np.array([[740, 20], [430, 1050], [1830, 1050], [1360, 30]])
    mask = np.zeros_like(frame_copy, dtype=np.uint8)
    cv2.fillPoly(mask, [trapezoid_pts], (255, 255, 255))
    inv_mask = cv2.bitwise_not(mask)
    black_overlay = np.zeros_like(frame_copy, dtype=np.uint8)
    alpha = 0.5 #투명도
    overlay = cv2.addWeighted(cv2.bitwise_and(black_overlay, inv_mask), alpha,
                            cv2.bitwise_and(frame_copy, inv_mask), 1 - alpha, 0)
    frame_copy = cv2.bitwise_or(overlay, cv2.bitwise_and(frame_copy, mask))
    

    # 첫번째 프레임에서 라인 검출한 것을 전체 프레임에 적용하기 위해
    if first_frame:
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY) #그레이스케일 변경
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 노이즈 제거
        edges = cv2.Canny(blurred, 50, 150) #엣지 검출
        edges_masked = cv2.bitwise_and(edges, edges, mask=border_mask) # 엣지 적용 마스킹 진행

        #라인 검출
        lines = cv2.HoughLinesP(edges_masked, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=30)

        h, w = frame.shape[:2]

        # HSB를 이용해 흰색 라인 검출
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])
        upper_white = np.array([180, 60, 255])

        # 라인이 검출되지 않은 경우
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 사각형 영역 내부인지 확인, ROI 밖 영역의 라인들 제거
                p1_in = cv2.pointPolygonTest(trapezoid_pts.astype(np.float32), (float(x1), float(y1)), False) >= 0
                p2_in = cv2.pointPolygonTest(trapezoid_pts.astype(np.float32), (float(x2), float(y2)), False) >= 0
                if not (p1_in and p2_in):
                    continue

                # 길이 필터링
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 150:
                    continue

                # 각도 계산
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

                # y 좌표 중 하나라도 980 이상 혹은 40보다 작으면 무시 진행
                if angle <= 30 and (y1 >= 990 or y2 >= 990) or (y1<40 or y2<40):
                    continue
                if angle > 30 and (y1 >= 990 or y2 >= 990) or (y1<40 or y2<40):
                    continue

                # ROI에서 흰색 비율 확인
                x_min = max(min(x1, x2), 0)
                y_min = max(min(y1, y2), 0)
                x_max = min(max(x1, x2), w - 1)
                y_max = min(max(y1, y2), h - 1)

                roi = hsv[y_min:y_max + 1, x_min:x_max + 1]
                if roi.size == 0:
                    continue

                # 해당 색상 범위 검출
                white_mask_roi = cv2.inRange(roi, lower_white, upper_white)
                white_ratio = cv2.countNonZero(white_mask_roi) / (roi.shape[0] * roi.shape[1] + 1e-5)
                if white_ratio <= 0.05: #흰색 비율이 제거한 경우
                    continue

                # 대표 x좌표, 기준 y좌표
                rep_x, rep_y = (x1, y1) if y1 < y2 else (x2, y2)
                # 기준이 되는 y가 더 큰 점의 x좌표
                y_base_x = x1 if y1 > y2 else x2

                # 검출된 라인들 결합
                detected_lines.append(((x1, y1, x2, y2), rep_x, angle, y_base_x))


        first_frame = False

    # 조건에 맞는 4개 선 필터링
    left_lines = [line for line in detected_lines if max(line[0][0], line[0][2]) < 900 and line[2] > 30]
    right_lines = [line for line in detected_lines if min(line[0][0], line[0][2]) >= 900 and line[2] > 30]
    top_lines = [line for line in detected_lines if (line[2] <= 40 or line[2] >= 150)and (40<min(line[0][1], line[0][3]) < 100)]
    bottom_lines = [line for line in detected_lines if (line[2] <= 40 or line[2] >= 150 and min(line[0][1], line[0][3]) >= 900)]

    # 위쪽 선: y좌표 최소
    if top_lines:
        top_line = min(top_lines, key=lambda l: min(l[0][1], l[0][3]))
    else:
        top_line = None

    # 아래쪽 선: y좌표 최대
    if bottom_lines:
        bottom_line = max(bottom_lines, key=lambda l: max(l[0][1], l[0][3]))
    else:
        bottom_line = None

    # 왼쪽, 오른쪽 선 중 대표 선 선택: (y_base_x가 가장 큰 것)
    if left_lines:
        left_line = max(left_lines, key=lambda l: l[3])
    else:
        left_line = None

    if right_lines:
        right_line = max(right_lines, key=lambda l: l[3])
    else:
        right_line = None



    # 위쪽 선: y좌표 최소
    if top_lines:
        top_line = min(top_lines, key=lambda l: min(l[0][1], l[0][3]))
    else:
        top_line = None

    # 아래쪽 선: y좌표 최대
    if bottom_lines:
        bottom_line = max(bottom_lines, key=lambda l: max(l[0][1], l[0][3]))
    else:
        bottom_line = None

    # 왼쪽, 오른쪽 선 중 대표 선 선택: (y_base_x가 가장 큰 것)
    if left_lines:
        left_line = max(left_lines, key=lambda l: l[3])
    else:
        left_line = None

    if right_lines:
        right_line = max(right_lines, key=lambda l: l[3])
    else:
        right_line = None

    # 직선 방정식 계산 (m, b)
    lines_eq = {}
    for name, line in zip(["left", "right", "top", "bottom"], [left_line, right_line, top_line, bottom_line]):
        if line is None:
            lines_eq[name] = None
            continue
        x1, y1, x2, y2 = line[0]
        m, b = line_equation(x1, y1, x2, y2)
        lines_eq[name] = (m, b)

    # 교점 계산
    # 왼쪽-위, 위-오른쪽, 오른쪽-아래, 아래-왼쪽 순서로 교점
    if None not in (lines_eq["left"], lines_eq["top"], lines_eq["right"], lines_eq["bottom"]):
        pt_left_top = intersection(lines_eq["left"], lines_eq["top"])
        pt_top_right = intersection(lines_eq["top"], lines_eq["right"])
        pt_right_bottom = intersection(lines_eq["right"], lines_eq["bottom"])
        pt_bottom_left = intersection(lines_eq["bottom"], lines_eq["left"])

        pts = [pt_left_top, pt_top_right, pt_right_bottom, pt_bottom_left]

        # 교점이 모두 계산된 경우 사각형 그리기
        if all(pt is not None for pt in pts):
            for i in range(4):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % 4]

                # 연결하는 두 점을 기준으로 기울기 계산
                dx = x2 - x1
                dy = y2 - y1
                angle = abs(np.degrees(np.arctan2(dy, dx)))

                # 기울기 <= 30도 파랑, >30도 빨강
                #사이드 라인
                if angle > 40 and angle < 150:
                    color = (57, 255, 20)  # 형광 초록 (BGR)
                    cv2.line(frame_copy, (x1, y1), (x2, y2), color, 3)

                    if max(x1, x2) < 900:  # 빨간 선이 화면 왼쪽에 위치한 경우만
                        m, b = line_equation(x1, y1, x2, y2)
                        if m is not None:
                            # throwin 기준으로 사용할 선의 x 경계값 저장
                            throwin_line_x = max(x1, x2)  # 오른쪽 끝
                        left_m=m #왼쪽 기울기
                        left_b=b #왼쪽
                    else:
                        m, b = line_equation(x1, y1, x2, y2)
                        if m is not None:
                            # throwin 기준으로 사용할 선의 x 경계값 저장
                            throwin_line_x = max(x1, x2)  # 오른쪽 끝
                        right_m=m #왼쪽 기울기
                        right_b=b #왼쪽
                        
                #코너킥, 골라인
                # 빨간 선 그리기 및 노란 선 그리기
                else:
                    color = (255, 0, 255)  # 형광 보라 (BGR)
                    cv2.line(frame_copy, (x1, y1), (x2, y2), color, 3)

                    m, b = line_equation(x1, y1, x2, y2)
                    if m is not None:
                        x_mid = (x1 + x2) // 2
                        offset = 100 if max(y1, y2) >= 800 else 50
                        x_left = x_mid - offset
                        x_right = x_mid + offset
                        y_left = int(m * x_left + b)
                        y_right = int(m * x_right + b)

                        # 위의 점 좌표 추가
                        h, w = frame.shape[:2]
                        if 0 <= x_left < w and 0 <= x_right < w and 0 <= y_left < h and 0 <= y_right < h:
                            cv2.line(frame_copy, (x_left, y_left), (x_right, y_right), (0, 255, 255), 4)  # 노란색
                            if y_left <100 or y_right <100:
                                yellow_line_top=(x_left, x_right, y_left, y_right)
                            yellow_line = (x_left, x_right, y_left, y_right)
    
    frame_count += 1 #프레임 수 증가
    current_players = {}
    
    # 사람을 찾기 위해, 사람을 누적해서 검출 진행
    results = model.track(source=frame, persist=True, conf=0.001, iou=0.5, classes=[0], tracker="bytetrack.yaml")
    #results = model.predict(frame, conf=0.25, iou=0.5, classes=[0])

    # 공을 찾기 위해, 공만 탐지하면 되기에 predict 사용
    results_ball = model.predict(frame, classes=[32], conf=0.001)

    # 사람을 찾은 경우
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else range(len(xyxy))

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            conf = confs[i]
            obj_id = ids[i]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 좌표가 1500보다 크면 제외, 축구장 밖에 있는 사람 제거하기 위해
            if cx > 1500 or cy > 1500:
                continue


            x1_clamped = max(0, x1)
            y1_clamped = max(0, y1)
            x2_clamped = min(frame.shape[1], x2)
            y2_clamped = min(frame.shape[0], y2)
            roi = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

            if roi.size == 0:
                continue

            # HSV 변경, 팀 색상 구분하기 위해
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 유니폼 색상에 따라 빨간팀, 파란팀 구분 진행
            # 선택할 빨간색의 범위
            red1 = cv2.inRange(hsv_roi, np.array([0, 40, 40]), np.array([20, 255, 255]))
            red2 = cv2.inRange(hsv_roi, np.array([160, 40, 40]), np.array([180, 255, 255]))
            magenta = cv2.inRange(hsv_roi, np.array([140, 40, 40]), np.array([160, 255, 255]))
            red_mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), magenta)
            red_ratio = np.count_nonzero(red_mask) / red_mask.size # 해당 사람이 빨간색을 포함하고 있는 비율

            #선택할 파란색의 범위
            blue_mask = cv2.inRange(hsv_roi, np.array([90, 70, 50]), np.array([130, 255, 255]))
            blue_ratio = np.count_nonzero(blue_mask) / blue_mask.size # 해당 사람이 파란색을 포함하고 있는 비율

            red_weight = 3.0 # 빨간팀 사람들이 잘 안 잡하기에 빨간 색에 가중치를 좀더 줌
            blue_weight = 2.0 # 파란팀 색생의 가중치
            red_weighted = red_ratio * red_weight
            blue_weighted = blue_ratio * blue_weight

            #팀을 구별할 threshold 값
            red_thresh = 0.01
            blue_thresh = 0.01

            # 팀 구별 진행
            if red_weighted > blue_weighted and red_weighted > red_thresh:
                team_color = (0, 0, 255)  # 빨강팀 (BGR)
                team_name = "Red Team"
            elif blue_weighted > red_weighted and blue_weighted > blue_thresh:
                team_color = (255, 0, 0)  # 파랑팀
                team_name = "Blue Team"
            else:
                team_color = (0, 255, 0)  # 알 수 없음 (초록)
                team_name = "Unknown"

            # 현재 선수들의 좌표 추출
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_players[obj_id] = (cx, cy)
            player_colors[obj_id] = team_color # 선수들의 팀 분류 진행

            # 속도 및 활동량 계산
            current_pos = (cx, cy)
            if obj_id in player_last_positions:
                prev_pos = player_last_positions[obj_id] # 기존에 움직인 값, 활동량을 표현하기 위해
                pixel_dist = euclidean(prev_pos, current_pos)
                real_dist_m = pixel_dist * pixel_to_meter # 움직인 거리에 픽셀 당 m 고려
                speed_kmh = (real_dist_m * fps) * 3.6 # 속도 변환 km/h 단위
                player_distances[obj_id] += real_dist_m #실제로 움직인 거리 산출

                # 이미지에 속도와 활동량 표현 진행
                cv2.putText(frame_copy, f"{speed_kmh:.1f} km/h", (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
                cv2.putText(frame_copy, f"Total: {player_distances[obj_id]:.1f} m", (cx, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

            player_last_positions[obj_id] = current_pos
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), team_color, 2)


    # 공 탐지 및 처리, 공의 trail 표현
    ball_detected = False  # 공 감지 여부

    # 공을 탐지한 경우
    if results_ball and results_ball[0].boxes is not None:
        boxes = results_ball[0].boxes
        cls = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # 공이라고 판단할 사이즈 선정
        min_ball_area = 50
        max_ball_area = 350

        for i, class_id in enumerate(cls):
            if class_id != 32: # cocodataset에서 공의 id는 32번임
                continue
            confidence = confs[i]
            if confidence < 0.001: # 신뢰도를 조정하여 공 검출 진행
                continue

            x1, y1, x2, y2 = map(int, xyxy[i])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # 경기장 밖에 있는 공은 탐지 X
            frame_center_x = frame.shape[1] // 2
            if abs(cx) < 300:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 탐지할 공의 색상, 흰색 여부 판단
            lower_white = np.array([0, 0, 240])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
            white_ratio = np.count_nonzero(white_mask) / white_mask.size

            # 흰색이 일정범위 이상 넘어선 경우 공이라고 판단 진행 X
            if white_ratio > 0.001:
                continue

            # 영상에서 공의 색상은 빨간색이라 빨간색이 존재하는 공 탐지 진행
            lower_red1 = np.array([0, 30, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 30, 70])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_ratio = np.count_nonzero(red_mask) / red_mask.size
            if red_ratio < 0.05: # 빨간색을 포함하지 않은 경우 공이라고 판단 X
                continue

            #공의 넓이 계산
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # 공의 넓이가 특정 범위보다 넘어선 경우 공이라고 판단 X
            if area < min_ball_area or area > max_ball_area:
                continue

            # 공 좌표 기록
            ball_positions.append((cx, cy))
            if len(ball_positions) > 10:
                ball_positions.pop(0)

            ball_detected = True  # 공 감지됨

            # 골 판정
            if yellow_line:
                # 골라인 좌표 (영상 아래 골대)
                x_left, x_right, y_left, y_right = yellow_line

                # 골라인 좌표 (영상 위 골대)
                x_left_top, x_right_top, y_left_top, y_right_top = yellow_line_top

                #해당 골라인에 들어간 경우 골이라고 판단 진행
                if x_left <= cx <= x_right:
                    if cy >= 800 and cy > max(y_left, y_right):
                        goal_flag = True # 골 flag
                        goal_frame_counter = goal_display_frames # 골 문자를 일정 프레임동안 출력하기 위해
                        red_team_goal_flag = True
                        goal_line_flag=False # 골라인 아웃 flag 초기화
                        throwin_flag=False # 스로인 flag 초기화
                    elif cy <= 100 and cy < min(y_left_top, y_right_top):
                        goal_flag = True # 골 flag
                        goal_line_flag=False # 골라인 아웃 flag 초기화
                        throwin_flag=False # 스로인 flag 초기화
                        goal_frame_counter = goal_display_frames # 골 문자를 일정 프레임동안 출력하기 위해
                        blue_team_goal_flag = True

                # 골 라인 아웃 판정
                if cx < x_left or cx > x_right:
                    if cy >= 800 and cy > max(y_left, y_right):
                        goal_line_flag = True # 골라인 아웃 flag
                        goal_line_out_counter = goal_display_frames # 골 라인 아웃을 일정 프레임동안 출력하기 위해
                        throwin_flag=False #스로인 flag 초기화

                    elif cy <= 100 and cy < min(y_left_top, y_right_top):
                        goal_line_flag = True # 골라인 아웃 flag
                        goal_line_out_counter = goal_display_frames # 골 아웃을 일정 프레임동안 출력하기 위해
                        throwin_flag=False #스로인 flag 초기화
            # throwin 판정
            # 해당 공이 사이드 라인을 벗어난 경우 골라인 아웃이라 판단 진행
            expected_x = int((cy - left_b) / left_m)
            expected_x_right = int((cy - right_b) / right_m)
            if (cx < expected_x) or ((cx > expected_x_right) and (goal_flag is False) and (goal_line_flag is False)):
                # 스로인 flag만 true
                throwin_flag = True
                goal_line_flag = False
                goal_flag = False


            # 거리 체크, 스로인 진행 여부를 판단하기 위해
            for player_pos in current_players.values():
                dist = np.linalg.norm(np.array(player_pos) - np.array((cx, cy)))
                if dist < 70: # 스로인 진행 시 공과 사람이 가까워 졌을 때
                    throwin_reset_pending = True
                    break
            else:
                throwin_reset_pending = False

            # 공과 사람이 멀어졌으며 공이 경기장 안으로 들어왔을 때 스로인 판단 초기화
            if throwin_flag and throwin_reset_pending and cx > expected_x:
                throwin_flag = False
                throwin_reset_pending = False

            break  # 첫 번째 유효 공만 처리

    # 항상 trail 그리기 (공 감지 여부와 무관)
    if ball_positions:
        # 여러 프레임 겹처서 trail 그리기
        overlay = frame_copy.copy()
        trail_len = len(ball_positions)
        for i in range(1, trail_len):
            pt1 = ball_positions[i - 1]
            pt2 = ball_positions[i]

            alpha = i / trail_len
            color = (0, 255, 255)
            thickness = max(5, int(4 * alpha))

            #투명도를 이용하여 공의 trail 표현 진행
            cv2.line(overlay, pt1, pt2, color, thickness)
            fade = alpha * 0.2
            frame_copy = cv2.addWeighted(overlay, fade, frame_copy, 1 - fade, 0) # 투명도 적용


    # GOAL 표시
    if goal_flag and goal_frame_counter > 0:
        # 골 문자 표현
        cv2.putText(frame_copy, "GOAL", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 102), 10)
        goal_frame_counter -= 1
        if goal_frame is None:
            goal_frame = frame_count
            print(f"start frame {goal_frame}")

        # 일정 시간 골 문자를 표현했을 때 초기화 진행
        if goal_frame_counter == 0:
            goal_flag = False
            goal_sound_played = False

    # 골을 넣을 시 음악 재생
    if goal_flag and not goal_sound_played and goal_line_flag==0:
        goal_sound.play()
        goal_sound_played = True

    # 골라인 아웃 표시
    if goal_line_flag and goal_line_out_counter > 0:
        cv2.putText(frame_copy, "GOAL LINE OUT", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
        goal_line_out_counter -= 1

            # print(f"start frame {goal_frame}")
        if goal_line_out_counter == 0:
            goal_line_flag = False


    # 스로인 표현  
    if throwin_flag:
        cv2.putText(frame_copy, "Throw-In", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 3, (57, 255, 20), 8)
    
    # 미니맵 크기 및 생성
    mini_map_width, mini_map_height = 520, 300
    mini_map = np.ones((mini_map_height, mini_map_width, 3), dtype=np.uint8) * 0
    mini_map[:] = (0, 180, 0)  # 녹색 배경

    # 라인의 색깔 및 두께
    line_color = (255, 255, 255)
    line_thickness = 2

    # 중앙선, 중앙원, 페널티 박스, 골대 그리기
    cv2.line(mini_map, (mini_map_width // 2, 0), (mini_map_width // 2, mini_map_height), line_color, line_thickness)
    cv2.circle(mini_map, (mini_map_width // 2, mini_map_height // 2), int(mini_map_height * 9.15 / 68), line_color, line_thickness)

    # 실제 패널티 박스 크기를 고려해 미니맵에 표현 진행
    penalty_box_length = int(mini_map_width * (16.5 / 105))
    penalty_box_height = int(mini_map_height * (40.3 / 68))

    # 패널티 박스 
    left_top = (0, (mini_map_height - penalty_box_height) // 2)
    left_bottom = (penalty_box_length, (mini_map_height + penalty_box_height) // 2)
    cv2.rectangle(mini_map, left_top, left_bottom, line_color, line_thickness)

    right_top = (mini_map_width - penalty_box_length, (mini_map_height - penalty_box_height) // 2)
    right_bottom = (mini_map_width, (mini_map_height + penalty_box_height) // 2)
    cv2.rectangle(mini_map, right_top, right_bottom, line_color, line_thickness)

    # 골대 크기, 실제 골대 크기의 비율을 고려해 적용
    goal_width = int(mini_map_height * (7.32 / 68))
    goal_thickness = 4

    # 미니맵에 골대 크기 적용
    left_goal_y1 = (mini_map_height // 2) - (goal_width // 2)
    left_goal_y2 = (mini_map_height // 2) + (goal_width // 2)
    cv2.line(mini_map, (2, left_goal_y1), (2, left_goal_y2), line_color, goal_thickness)

    right_goal_y1 = (mini_map_height // 2) - (goal_width // 2)
    right_goal_y2 = (mini_map_height // 2) + (goal_width // 2)
    cv2.line(mini_map, (mini_map_width - 3, right_goal_y1), (mini_map_width - 3, right_goal_y2), line_color, goal_thickness)

    # Perspective 변환을 위한 포인트 설정 
        # 원본 이미지에서 미니맵으로 변환할 원근 변환을 위한 기준점 설정
    src_pts = np.float32([
        [750, 40],    # 좌상단 점 (경기장 원본 좌표 기준)
        [480, 995],   # 좌하단 점
        [1760, 995],  # 우하단 점
        [1350, 40]    # 우상단 점
    ])

    # 미니맵 상에 대응되는 목적 좌표 (직사각형 형태로 정규화)
    dst_pts = np.float32([
        [0, 0],                             # 좌상단
        [mini_map_width, 0],               # 우상단
        [mini_map_width, mini_map_height], # 우하단
        [0, mini_map_height]               # 좌하단
    ])

    # Perspective Transform 행렬 계산
    # src_pts의 4점을 dst_pts의 4점으로 매핑하는 변환 행렬 M 생성
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 현재 프레임에서 감지된 모든 선수 위치를 미니맵에 표시
    for pid, pos in current_players.items():
        # 현재 선수 위치를 배열 형태로 변환
        pt = np.array([[[pos[0], pos[1]]]], dtype=np.float32)

        # 선수 위치에 원근 변환 적용하여 미니맵 좌표로 변환
        map_pt = cv2.perspectiveTransform(pt, M)
        map_x, map_y = int(map_pt[0, 0, 0]), int(map_pt[0, 0, 1])

        # 선수의 팀 또는 고유 색상 지정
        color = player_colors.get(pid, (255, 255, 255))

        # 미니맵은 y축이 아래로 갈수록 증가하므로 flip 처리
        map_y = mini_map_height - map_y

        # 미니맵에 원으로 선수 표시
        cv2.circle(mini_map, (map_x, map_y), 4, color, -1)

    # 공의 위치를 미니맵에 표시 (trail의 가장 마지막 위치 사용)
    if ball_positions:
        bx, by = ball_positions[-1]  # 가장 최근 프레임의 공 위치

        # 공 좌표를 배열로 변환
        ball_pt = np.array([[[bx, by]]], dtype=np.float32)

        # 공 위치에도 동일하게 원근 변환 적용
        ball_map_pt = cv2.perspectiveTransform(ball_pt, M)
        map_x, map_y = int(ball_map_pt[0, 0, 0]), int(ball_map_pt[0, 0, 1])

        # y축 뒤집기 처리
        map_y = mini_map_height - map_y

        # 미니맵에 공 표시 (노란색 원)
        cv2.circle(mini_map, (map_x, map_y), 5, (0, 255, 255), -1)



    # 미니맵 테두리
    cv2.rectangle(mini_map, (0, 0), (mini_map_width - 1, mini_map_height - 1), line_color, 2)

    # 미니맵 프레임 왼쪽 상단에 붙임
    frame_copy[10:10 + mini_map_height, 10:10 + mini_map_width] = mini_map


        # 볼 점유율 계산용 변수 초기화
    if 'ball_possession_count' not in globals():
        ball_possession_count = {'Red Team': 0, 'Blue Team': 0, 'Unknown': 0}
        ball_owner_distance_thresh = 100  # 공과 선수 간 거리 임계값(px)


        # 3. 점유율 계산
        # 공의 가장 최근 위치 가져오기 (trail에서 마지막 좌표)
    ball_center = ball_positions[-1] if ball_positions else None

    # 공 근처에 있는 팀과 거리 초기화
    nearest_team = None  # 공 근처에 있는 팀(Red Team 또는 Blue Team)
    nearest_dist = float('inf')  # 최소 거리 초기값 (무한대)

    # 공이 감지된 경우에만 처리
    if ball_center:
        # 현재 프레임에 있는 모든 선수에 대해 반복
        for pid, pos in current_players.items():
            # 공과 선수 간 거리 계산 (유클리드 거리)
            dist = euclidean(ball_center, pos)
            # 지금까지 중 가장 가까운 선수이면서, 특정 거리 기준 안일 경우
            if dist < nearest_dist and dist < ball_owner_distance_thresh:
                nearest_dist = dist  # 최소 거리 업데이트

                # 선수의 고유 색상(팀 색상)을 가져옴
                team_color = player_colors.get(pid)

                # 색상으로부터 소속 팀 결정
                if team_color == (0, 0, 255):        # 빨간색 -> Red Team
                    nearest_team = 'Red Team'
                elif team_color == (255, 0, 0):      # 파란색 -> Blue Team
                    nearest_team = 'Blue Team'

    # 공 근처에 팀이 감지된 경우에만 해당 팀의 점유 횟수 누적
    if nearest_team:
        ball_possession_count[nearest_team] += 1

    # 항상 점유율을 계산 (total이 0인 경우 0으로 처리)
    total = sum(ball_possession_count.values())
    red_pct = (ball_possession_count['Red Team'] / total) * 100 if total else 0
    blue_pct = (ball_possession_count['Blue Team'] / total) * 100 if total else 0


    # 6. 시리얼 전송 (항상 시도)
    if ser:
        try:
            # 1. 이벤트 전송 우선 (변화 있을 때만)
            current_event = ""
            if goal_flag:
                current_event = "GOAL"
            elif throwin_flag:
                current_event = "THROWIN"
            elif goal_line_flag:
                current_event = "OUT"

            if current_event != "" :
                ser.write(f"{current_event}\n".encode())

            # 2. 점유율 전송 (항상 또는 주기적 혹은 변화가 있을 때만)
            current_possession = f"{int(red_pct)},{int(blue_pct)}"
            if current_possession != last_possession_sent:
                ser.write(f"{current_possession}\n".encode())
                last_possession_sent = current_possession

        except serial.SerialException:
            print("시리얼 전송 실패")




    # 글씨 스타일 공통 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2

    # 프레임 번호와 FPS 기반 시간 계산
    time_elapsed = frame_count / fps 
    minutes = int(time_elapsed // 60)
    seconds = int(time_elapsed % 60)
    time_str = f"{minutes}:{seconds:02d}"

    # 시간 박스 (흰 배경 -> 검은 글씨)
    time_text = f"Time: {time_str}"
    (text_w, text_h), baseline = cv2.getTextSize(time_text, font, scale, thickness)
    cv2.rectangle(frame_copy, (10, 400 - text_h - 10), (10 + text_w + 10, 400 + baseline + 10), (255, 255, 255), -1)
    cv2.putText(frame_copy, time_text, (10 + 5, 400), font, scale, (0, 0, 0), thickness)

    # 레드 점유율 박스 (흰 배경 -> 빨간 글씨)
    red_text = f"Red Possession: {red_pct:.1f}%"
    (text_w, text_h), baseline = cv2.getTextSize(red_text, font, scale, thickness)
    cv2.rectangle(frame_copy, (10, 440 - text_h - 10), (10 + text_w + 10, 440 + baseline + 10), (255, 255, 255), -1)
    cv2.putText(frame_copy, red_text, (10 + 5, 440), font, scale, (0, 0, 255), thickness)

    # 블루 점유율 박스 (흰 배경 -> 파란 글씨)
    blue_text = f"Blue Possession: {blue_pct:.1f}%"
    (text_w, text_h), baseline = cv2.getTextSize(blue_text, font, scale, thickness)
    cv2.rectangle(frame_copy, (10, 480 - text_h - 10), (10 + text_w + 10, 480 + baseline + 10), (255, 255, 255), -1)
    cv2.putText(frame_copy, blue_text, (10 + 5, 480), font, scale, (255, 0, 0), thickness)


    cv2.imshow("Soccer Tracking", frame_copy) #현재 이미지 출력
    out.write(frame_copy)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 종료
        break

pygame.mixer.stop()
pygame.mixer.quit()

cap.release()
out.release()
cv2.destroyAllWindows()  


# 골이 발생한 경우에만 하이라이트 영상 저장 및 전송을 수행
if goal_frame is not 0:
    highlight_margin = 150  # 골 주변 전후 프레임 수 (약 5초 전후)

    # 기존 영상 파일 열기
    cap = cv2.VideoCapture(output_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 영상 너비
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 영상 높이
    fps = cap.get(cv2.CAP_PROP_FPS)                         # 프레임 속도
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 총 프레임 수

    if goal_frame is not None:
        # 시작/끝 프레임 계산 (영상 범위 초과 방지)
        start_frame = max(0, goal_frame - highlight_margin)
        end_frame = min(total_frames - 1, goal_frame + highlight_margin)

        # 하이라이트 영상 저장을 위해 기존에 이미지 처리한 영상을 다시 출력 진행
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            print("비디오 열기 실패")
            exit()

        # 하이라이트 영상 저장
        out = cv2.VideoWriter(
            "./output_video/highlight.mp4",                   # 저장 경로
            cv2.VideoWriter_fourcc(*"mp4v"),                  # 코덱 설정
            fps,                                              # 프레임 속도
            (frame_width, frame_height)                       # 영상 해상도
        )

        current_frame = 0  # 현재 프레임 인덱스
        while cap.isOpened():
            ret, frame = cap.read()  # 프레임 읽기
            if not ret:
                break
            current_frame += 1
            # 하이라이트 범위 내 프레임만 저장
            if current_frame <= end_frame and current_frame >= start_frame:
                out.write(frame)

        cap.release()
        out.release()
        print("highlight.mp4 저장 완료!")


    # 이메일 전송 설정
    import smtplib
    from email.message import EmailMessage

    # 발신자 이메일
    sender_email = "????@handong.ac.kr"
    receiver_email = None  # 수신자 이메일은 골 상황에 따라 설정

    # 골 상황에 따라 수신자 결정
    if red_team_goal_flag == 1:
        receiver_email = "????@gmail.com"
    elif blue_team_goal_flag == 1:
        receiver_email = "????@handong.ac.kr"

    # 이메일 제목 및 본문
    subject = "Highlight Video"
    body = "첨부된 영상은 하이라이트 영상입니다."

    # Gmail 앱 비밀번호 (2단계 인증 사용 시 앱 비밀번호 필요), 사용자 설정에 따라 변경이 필요함
    password = "???? ???? ???? ????"

    # 수신자가 설정된 경우에만 이메일 전송
    if receiver_email:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.set_content(body)

        # 하이라이트 영상 첨부
        file_path = "./output_video/highlight.mp4"
        with open(file_path, "rb") as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype="video", subtype="mp4", filename="highlight.mp4")

        # Gmail SMTP 서버를 통해 이메일 전송
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)

        print("이메일 전송 완료!")
    else:
        print("goal flag가 감지되지 않아 이메일을 전송하지 않았습니다.")
