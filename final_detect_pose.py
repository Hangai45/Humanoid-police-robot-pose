import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque, Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# PATHS (чинийх)
# =========================
pose_model = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\pose_landmarker_full.task"
hand_model = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\hand_landmarker.task"

FEATURES_DIR = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\version_2\dataset\features"
# Нэгтгэсэн файл:
FEATURES_MERGED = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\version_2\dataset\merged_features.csv"
# Хуучин fallback
FEATURES_FALLBACK = r"C:\Users\DELL.DESKTOP-PTQ10MO\Desktop\WEB\police robot\version_2\dataset\features.csv"

# =========================
# FEATURE ORDER (таны recorder-той яг адил)
# =========================
POSE_ORDER = ["LS", "RS", "LE", "RE"]
FINGER_ORDER = ["T", "I", "M", "R", "P"]

# =========================
# ANGLE JOINTS (таны код)
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

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y], dtype=np.float32)
    v2 = np.array([p3.x - p2.x, p3.y - p2.y], dtype=np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    v1 /= n1; v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(dot)))
    return round(ang, 1)

def get_pose_angles(pose_result):
    angles = {}
    if not pose_result.pose_landmarks:
        return angles
    lm = pose_result.pose_landmarks[0]
    for name, (i1, i2, i3) in ANGLE_JOINTS.items():
        try:
            angles[name] = calculate_angle(lm[i1], lm[i2], lm[i3])
        except:
            angles[name] = 0.0
    return angles

def get_hand_angles(hand_result):
    hand_angles = {}
    if not hand_result.hand_landmarks:
        return hand_angles

    # ⚠️ таны recorder: hand_idx==0 -> "L", hand_idx==1 -> "R"
    # (зарим үед солигдож болно, гэхдээ dataset чинь үүнийг дагаж цугларсан тул ижил байлгана)
    for hand_idx, hand_lm in enumerate(hand_result.hand_landmarks):
        hand_label = "L" if hand_idx == 0 else "R"
        hand_angles[hand_label] = {}
        for finger, (i1, i2, i3) in HAND_ANGLE_JOINTS.items():
            try:
                hand_angles[hand_label][finger] = calculate_angle(hand_lm[i1], hand_lm[i2], hand_lm[i3])
            except:
                hand_angles[hand_label][finger] = 0.0
    return hand_angles

def feature_vector(pose_res, hand_angles):
    """
    14 angles + 4 vertical diffs = 18 features
    """
    if not pose_res or not pose_res.pose_landmarks:
        return np.zeros(18, dtype=np.float32)
        
    lm = pose_res.pose_landmarks[0]
    pose_angles = get_pose_angles(pose_res)
    
    # -- 1. Angles (4) --
    feats = [float(pose_angles.get(k, 0.0)) for k in POSE_ORDER]
    
    # -- 2. Hand Angles (10) --
    for side in ["R","L"]:
        if side in hand_angles:
            feats += [float(hand_angles[side].get(k, 0.0)) for k in FINGER_ORDER]
        else:
            feats += [0.0]*5
            
    # -- 3. Vertical Diffs (4) --
    # RW_Y: Right Wrist - Elbow
    # LW_Y: Left Wrist - Elbow
    # RE_Y: Right Elbow - Shoulder
    # LE_Y: Left Elbow - Shoulder
    rw_y = round(float(lm[16].y - lm[14].y), 4)
    lw_y = round(float(lm[15].y - lm[13].y), 4)
    re_y = round(float(lm[14].y - lm[12].y), 4)
    le_y = round(float(lm[13].y - lm[11].y), 4)
    
    feats += [rw_y, lw_y, re_y, le_y]
    
    return np.array(feats, dtype=np.float32)  # (18,)

# =========================
# DATASET LOAD
# =========================
def load_dataset():
    # 1. Эхлээд нэгтгэсэн файлыг хайна
    if os.path.exists(FEATURES_MERGED):
        print(f"Loading merged file: {FEATURES_MERGED}")
        return pd.read_csv(FEATURES_MERGED, encoding="utf-8")

    # 2. Олдохгүй бол features/ доторх бүх файлыг уншина
    csvs = []
    if os.path.isdir(FEATURES_DIR):
        csvs = glob.glob(os.path.join(FEATURES_DIR, "*.csv"))
        xlsx_files = glob.glob(os.path.join(FEATURES_DIR, "*.xlsx"))
        csvs.extend(xlsx_files)
        
    if not csvs and os.path.exists(FEATURES_FALLBACK):
        csvs = [FEATURES_FALLBACK]

    if not csvs:
        raise FileNotFoundError("Data олдсонгүй. Эхлээд data цуглуулж эсвэл combine_data.py ажиллуулна уу.")

    dfs = []
    for p in csvs:
        try:
            if p.endswith(".csv"):
                df = pd.read_csv(p, encoding="utf-8")
            else:
                df = pd.read_excel(p)
                
            if "label" in df.columns:
                dfs.append(df)
        except Exception as e:
            print("Skip:", os.path.basename(p), "err:", e)

    if not dfs:
        raise RuntimeError("CSV/Excel-үүд уншигдсангүй эсвэл 'label' багана алга.")

    return pd.concat(dfs, ignore_index=True)

def to_4class(label: str) -> str:
    s = str(label).lower()
    if s == "raise_right_hand" or s == "right":
        return "right"
    if s == "raise_left_hand" or s == "left":
        return "left"
    if s == "both_hands_up" or s == "both":
        return "both"
    return "other"

def get_feature_columns(df: pd.DataFrame):
    # recorder-ийнх: timestamp,label,burst_id,frame_idx + 18 feat
    base_cols = ["timestamp", "label", "burst_id", "frame_idx"]
    feat_cols = [c for c in df.columns if c not in base_cols]
    # Хэрвээ баганын нэр өөр байвал сүүлчийн 18-г авч болно
    if len(feat_cols) < 18 and df.shape[1] >= 22:
        feat_cols = list(df.columns[-18:])
    return feat_cols

# =========================
# TRAIN MODEL
# =========================
print("Loading dataset...")
df = load_dataset()
feat_cols = get_feature_columns(df)

# X, y
X = df[feat_cols].astype(np.float32).values
y = df["label"].apply(to_4class).values

print("Dataset shape:", X.shape, "labels:", Counter(y))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    ))
])

print("Training...")
model.fit(X_train, y_train)

print("\nEval:")
pred = model.predict(X_test)
print(classification_report(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# =========================
# MEDIAPIPE SETUP (VIDEO)
# =========================
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

pose_landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model),
        running_mode=VisionRunningMode.VIDEO
    )
)
hand_landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2
    )
)

# =========================
# REALTIME LOOP
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera нээгдсэнгүй. VideoCapture(0/1) шалгаарай.")

# smoothing (сүүлийн 12 prediction)
HIST = deque(maxlen=12)

# “other” гэж гарч байвал, right/left гэж баталгаажуулах threshold
PROB_THRESHOLD = 0.50  # 4 классд 0.50 хангалттай

print("\nRealtime ON (q=quit)\n")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        ts = int(time.time() * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_res = pose_landmarker.detect_for_video(mp_image, ts)
        hand_res = hand_landmarker.detect_for_video(mp_image, ts)

        pose_angles = get_pose_angles(pose_res)
        hand_angles = get_hand_angles(hand_res)
        feats = feature_vector(pose_res, hand_angles).reshape(1, -1)

        # predict + proba
        proba = None
        label = "other"
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feats)[0]
            classes = list(model.named_steps["clf"].classes_)
            best_i = int(np.argmax(proba))
            best_label = classes[best_i]
            best_p = float(proba[best_i])

            # confidence gate — дохио таньсан бол
            if best_label in ("right", "left", "both") and best_p >= PROB_THRESHOLD:
                label = best_label
            else:
                label = "other"
        else:
            label = model.predict(feats)[0]

        HIST.append(label)
        stable = Counter(HIST).most_common(1)[0][0]

        # UI өнгө
        color_map = {
            "right": (0, 255, 0),    # ногоон
            "left":  (255, 165, 0),  # улбар шар
            "both":  (0, 200, 255),  # цэнхэр
            "other": (100, 100, 100) # саарал
        }
        clr = color_map.get(stable, (200, 200, 200))

        # UI текст
        text = f"Pred: {label}  |  Stable: {stable}"
        if proba is not None:
            classes = list(model.named_steps["clf"].classes_)
            pr = {c: float(proba[i]) for i, c in enumerate(classes)}
            text2 = (f"R={pr.get('right',0):.2f}  "
                     f"L={pr.get('left',0):.2f}  "
                     f"B={pr.get('both',0):.2f}  "
                     f"O={pr.get('other',0):.2f}")
            cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, clr, 2)

        cv2.imshow("Gesture Recognition — 4 класс", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
finally:
    pose_landmarker.close()
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
