import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import playsound # 알림 소리를 재생하기 위한 라이브러리

# --- 1. EAR (Eye Aspect Ratio) 계산 함수 ---
def eye_aspect_ratio(eye):
    # 눈의 수직 랜드마크 간의 유클리드 거리를 계산합니다.
    # A: (p2, p6), B: (p3, p5)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 눈의 수평 랜드마크 간의 유클리드 거리를 계산합니다.
    # C: (p1, p4)
    C = dist.euclidean(eye[0], eye[3])

    # EAR 공식을 적용합니다.
    ear = (A + B) / (2.0 * C)
    return ear

# --- 2. 상수 및 변수 설정 ---
# 졸음 감지 임계값 (눈이 감겼다고 판단하는 EAR 값)
EYE_AR_THRESH = 0.25
# 졸음으로 판단하기 위해 눈이 감겨 있어야 하는 프레임 수
EYE_AR_CONSEC_FRAMES = 48 # 약 1.5 ~ 2초 (프레임 속도에 따라 다름)

# 카운터와 알람 플래그 초기화
COUNTER = 0
ALARM_ON = False

# 랜드마크 예측기 초기화 및 얼굴 랜드마크 인덱스 추출
print("[INFO] 얼굴 랜드마크 예측기 로딩...")
# 'shape_predictor_68_face_landmarks.dat' 파일 경로를 지정해야 합니다.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
detector = dlib.get_frontal_face_detector()

# 68개 랜드마크 중 왼쪽 눈과 오른쪽 눈의 인덱스를 추출
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# --- 3. 웹캠 스트림 시작 ---
print("[INFO] 웹캠 스트림 시작...")
cap = cv2.VideoCapture(0)
time.sleep(1.0) # 웹캠이 시작될 때까지 잠시 대기

# 알림 소리 파일 경로 (적절한 WAV/MP3 파일로 대체 필요)
# 예를 들어, 'alarm.wav'와 같은 파일을 사용하세요.
ALARM_SOUND_PATH = "alarm.wav" 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지 (1은 업샘플링 횟수)
    rects = detector(gray, 0)

    # 감지된 얼굴에 대해 반복
    for rect in rects:
        # 얼굴 랜드마크 좌표 예측 및 NumPy 배열로 변환
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 왼쪽/오른쪽 눈 좌표를 추출하고 EAR 계산
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 두 눈의 평균 EAR 계산
        ear = (leftEAR + rightEAR) / 2.0
        
        # 눈곽을 그려 시각화 (선택 사항)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # --- 4. 졸음 감지 로직 ---
        if ear < EYE_AR_THRESH:
            # 눈이 감긴 상태가 지속될 경우 카운터 증가
            COUNTER += 1
            
            # 눈이 감긴 프레임 수가 임계값을 초과하면 알람 활성화
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # 알람이 켜져 있지 않은 경우 알람 활성화 (중복 방지)
                if not ALARM_ON:
                    ALARM_ON = True
                    # 별도의 스레드로 알람 소리 재생
                    playsound.playsound(ALARM_SOUND_PATH)
                    print("[ALARM] 졸음 감지!")
                
                # 화면에 경고 문구 표시
                cv2.putText(frame, "DANGER: DROWSINESS DETECTED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        else:
            # 눈이 다시 뜨인 경우 카운터와 알람 상태 리셋
            COUNTER = 0
            ALARM_ON = False

        # 현재 EAR 값을 화면에 표시
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
 
    # 결과 화면 표시
    cv2.imshow("Drowsiness Detector", frame)
    
    # 'q' 키를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 종료 시 정리 작업
cv2.destroyAllWindows()
cap.release()