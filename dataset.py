import os
import gc
import csv
import time
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

# =========================
# PATHS
# =========================
pose_model = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\pose_landmarker_full.task"
hand_model = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\hand_landmarker.task"

DATASET_DIR   = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\version_2\dataset"
FEATURES_DIR  = os.path.join(DATASET_DIR, "features")   # бурст тус бүр тусдаа файл

# =========================
# LABELS
# =========================
CLASSES = [
    "raise_right_hand",   # баруун гар өргөх
    "raise_left_hand",    # зүүн гар өргөх
    "both_hands_up",      # 2 гар өргөх
    "other",              # дохиогүй / бусад
]

BURST_COUNT = 1000
BURST_SKIP  = 1

os.makedirs(FEATURES_DIR, exist_ok=True)

# =========================
# MEDIAPIPE SETUP
# =========================
BaseOptions        = mp.tasks.BaseOptions
VisionRunningMode  = mp.tasks.vision.RunningMode
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# Сильна reference — GC устгахаас хамгаална
_pose_opts = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model),
    running_mode=VisionRunningMode.IMAGE
)
_hand_opts = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)
pose_landmarker = PoseLandmarker.create_from_options(_pose_opts)
hand_landmarker = HandLandmarker.create_from_options(_hand_opts)

# =========================
# ANGLE JOINTS
# =========================
ANGLE_JOINTS = {
    "LS": (11, 13, 15),
    "RS": (12, 14, 16),
    "LE": (13, 15, 17),
    "RE": (14, 16, 18),
}
HAND_ANGLE_JOINTS = {
    "T": (0, 2, 4),
    "I": (0, 6, 8),
    "M": (0, 10, 12),
    "R": (0, 14, 16),
    "P": (0, 18, 20),
}

# =========================
# SKELETON CONNECTIONS
# =========================
POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),
    (11,12),(11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# =========================
# FUNCTIONS
# =========================
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y], dtype=np.float32)
    v2 = np.array([p3.x - p2.x, p3.y - p2.y], dtype=np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    v1 /= n1; v2 /= n2
    return round(float(np.degrees(np.arccos(float(np.clip(np.dot(v1, v2), -1.0, 1.0))))), 1)

def get_pose_angles(pose_res):
    angles = {}
    if not pose_res.pose_landmarks:
        return angles
    lm = pose_res.pose_landmarks[0]
    for name, (i1, i2, i3) in ANGLE_JOINTS.items():
        try:    angles[name] = calculate_angle(lm[i1], lm[i2], lm[i3])
        except: angles[name] = 0.0
    return angles

def get_hand_angles(hand_res):
    hand_angles = {}
    if not hand_res.hand_landmarks:
        return hand_angles
    for idx, hand_lm in enumerate(hand_res.hand_landmarks):
        label = "L" if idx == 0 else "R"
        hand_angles[label] = {}
        for finger, (i1, i2, i3) in HAND_ANGLE_JOINTS.items():
            try:    hand_angles[label][finger] = calculate_angle(hand_lm[i1], hand_lm[i2], hand_lm[i3])
            except: hand_angles[label][finger] = 0.0
    return hand_angles

def feature_vector(pose_res, hand_angles):
    """
    14 angles + 4 vertical diffs = 18 features
    """
    if not pose_res or not pose_res.pose_landmarks:
        return [0.0] * 18
        
    lm = pose_res.pose_landmarks[0]
    pose_angles = get_pose_angles(pose_res)
    
    # -- 1. Angles (4) --
    feats = [float(pose_angles.get(k, 0.0)) for k in ["LS","RS","LE","RE"]]
    
    # -- 2. Hand Angles (10) --
    for side in ["R","L"]:
        if side in hand_angles:
            feats += [float(hand_angles[side].get(k, 0.0)) for k in ["T","I","M","R","P"]]
        else:
            feats += [0.0]*5
            
    # -- 3. Vertical Diffs (4) -- (Y тэнхлэгийн зөрөө)
    # RW_Y: Right Wrist - Elbow
    # LW_Y: Left Wrist - Elbow
    # RE_Y: Right Elbow - Shoulder
    # LE_Y: Left Elbow - Shoulder
    # MediaPipe Y: Дээшээ байх тусам бага (0.0), доошоо байх тусам их (1.0)
    rw_y = round(float(lm[16].y - lm[14].y), 4)
    lw_y = round(float(lm[15].y - lm[13].y), 4)
    re_y = round(float(lm[14].y - lm[12].y), 4)
    le_y = round(float(lm[13].y - lm[11].y), 4)
    
    feats += [rw_y, lw_y, re_y, le_y]
    
    return feats  # len=18

def draw_pose(frame, pose_res):
    if not pose_res.pose_landmarks:
        return
    h, w = frame.shape[:2]
    lm = pose_res.pose_landmarks[0]
    for p in lm:
        cv2.circle(frame, (int(p.x*w), int(p.y*h)), 4, (0,255,255), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(lm) and b < len(lm):
            cv2.line(frame, (int(lm[a].x*w),int(lm[a].y*h)),
                            (int(lm[b].x*w),int(lm[b].y*h)), (0,200,255), 2)

def draw_hands(frame, hand_res):
    if not hand_res.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_lm in hand_res.hand_landmarks:
        pts = [(int(p.x*w), int(p.y*h)) for p in hand_lm]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (255,0,200), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255,255,0), -1)

# =========================
# CSV
# =========================
FEAT_COLS = [
    "LS","RS","LE","RE",
    "RH_T","RH_I","RH_M","RH_R","RH_P",
    "LH_T","LH_I","LH_M","LH_R","LH_P",
    "RW_Y", "LW_Y", "RE_Y", "LE_Y"
]

# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera нээгдсэнгүй!")

current_index = 0
current_label = CLASSES[current_index]
burst_active  = False
burst_saved   = 0
burst_id      = None
burst_label   = None
burst_csv_path = ""
skip_counter  = 0
total_rows    = 0
frame_count   = 0

# detection-г 2 frame тутамд хийх
DETECT_EVERY = 5   # 3 frame тутамд detect → хурдан, lag бага
pose_res = None
hand_res = None

print("="*50)
print("DATASET RECORDER — 4 класс")
print("="*50)
for i, c in enumerate(CLASSES):
    print(f"  {i}: {c}")
print("  n=дараагийн  SPACE=хадгалах  q=гарах")
print("="*50)

gc.disable()   # GC — callback crash-ийг хамгаалах

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # detection-г DETECT_EVERY frame тутамд л хийнэ (хурдасгах)
        if frame_count % DETECT_EVERY == 0:
            # жижиг resolution дээр detect → хурдан
            small = cv2.resize(frame, (640, 360))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                pose_res = pose_landmarker.detect(mp_img)
                hand_res = hand_landmarker.detect(mp_img)
            except Exception as e:
                print("Detection err:", e)

        if pose_res is None or hand_res is None:
            cv2.imshow("Dataset Recorder", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        pose_angles = get_pose_angles(pose_res)
        hand_angles = get_hand_angles(hand_res)
        feats = feature_vector(pose_res, hand_angles)

        # ---- burst saving ----
        if burst_active:
            skip_counter += 1
            if skip_counter >= BURST_SKIP:
                skip_counter = 0
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                try:
                    with open(burst_csv_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([ts, burst_label, burst_id, burst_saved] + feats)
                    burst_saved += 1
                    total_rows  += 1
                except Exception as e:
                    print("CSV err:", e)

                if burst_saved >= BURST_COUNT:
                    burst_active = False
                    burst_saved  = 0
                    print(f"✓ Дууслаа → {os.path.basename(burst_csv_path)}  Нийт rows: {total_rows}")

        # ---- skeleton ----
        draw_pose(frame, pose_res)
        draw_hands(frame, hand_res)

        # ---- UI ----
        cv2.putText(frame, f"Label: {current_label}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if burst_active:
            cv2.putText(frame, f"Saving '{burst_label}'... {burst_saved}/{BURST_COUNT}",
                        (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        else:
            cv2.putText(frame, "SPACE=эхлэх  n=солих  q=гарах",
                        (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(frame, f"Rows: {total_rows}",
                    (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow("Dataset Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('n') and not burst_active:
            current_index = (current_index + 1) % len(CLASSES)
            current_label = CLASSES[current_index]
            print("Label:", current_label)
        if key == 32 and not burst_active:
            burst_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
            burst_label   = current_label
            burst_csv_path = os.path.join(FEATURES_DIR, f"{burst_label}_{burst_id}.csv")
            # шинэ файлд header бичих
            with open(burst_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp","label","burst_id","frame_idx"] + FEAT_COLS)
            burst_active  = True
            burst_saved   = 0
            skip_counter  = 0
            print(f"[Start] label={burst_label}  file={os.path.basename(burst_csv_path)}")

finally:
    # MediaPipe-ийг албан ёсоор хаах (exit crash арилгах)
    if 'pose_landmarker' in locals(): pose_landmarker.close()
    if 'hand_landmarker' in locals(): hand_landmarker.close()
    gc.enable()
    cap.release()
    cv2.destroyAllWindows()
    print("Saved:", FEATURES_DIR)

