import csv
import math
import os
import time
import threading
import urllib.request
from collections import deque
import cv2
import numpy as np
import serial
import sys
import select

# ==========================================
# 1. โหลดระบบ AI (MediaPipe Tasks API - ใช้กับ mediapipe เวอร์ชันใหม่ 0.10.30+ / 1.x)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

def ensure_model(path, url):
    """ดาวน์โหลดไฟล์โมเดลอัตโนมัติถ้ายังไม่มี (ครั้งแรกครั้งเดียว)"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"⬇️ กำลังดาวน์โหลดโมเดล {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)
    print(f"✅ ดาวน์โหลด {os.path.basename(path)} เสร็จแล้ว")

# เส้นเชื่อมจุดบนมือ 21 จุด (ใช้วาดโครงมือแทน drawing_utils ที่ถูกตัดออกไป)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # นิ้วโป้ง
    (0, 5), (5, 6), (6, 7), (7, 8),          # นิ้วชี้
    (5, 9), (9, 10), (10, 11), (11, 12),     # นิ้วกลาง
    (9, 13), (13, 14), (14, 15), (15, 16),   # นิ้วนาง
    (13, 17), (17, 18), (18, 19), (19, 20),  # นิ้วก้อย
    (0, 17)                                  # ฝ่ามือ
]

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)
    ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)

    # ปรับ Confidence ให้สูงถึง 80% (เน้นชัวร์ ไม่เน้นมั่ว)
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.8,
        min_tracking_confidence=0.8,
    )
    hands_detector = mp_vision.HandLandmarker.create_from_options(hand_options)

    face_options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=0.8,
    )
    face_detector = mp_vision.FaceDetector.create_from_options(face_options)

    print(f"🧠 เปิดใช้งานระบบ AI (MediaPipe {getattr(mp, '__version__', '?')} / Tasks API): ตรวจจับใบหน้าและมือ")
except Exception as e:
    error_msg = str(e)
    print("\n" + "="*60)
    print(f"❌ โหลด AI (MediaPipe) ไม่สำเร็จ! สาเหตุที่แท้จริงคือ:")
    print(f"   👉 {error_msg}")
    print("="*60)
    if "DLL load failed" in error_msg:
        print("🛠️ วิธีแก้: เครื่องของคุณขาดไฟล์ระบบ Windows (C++ Redistributable)")
        print("   ให้ดาวน์โหลดและติดตั้งจากลิงก์นี้: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("   ติดตั้งเสร็จแล้วให้ Restart คอมพิวเตอร์ 1 ครั้งครับ")
    elif "No module named" in error_msg and "mediapipe" in error_msg:
        print("🛠️ วิธีแก้: ตัวรัน Python หาไฟล์ที่ติดตั้งไม่เจอ (รันผิด Environment)")
        print("   python -m pip install mediapipe")
    elif "urlopen" in error_msg or "HTTP" in error_msg or "getaddrinfo" in error_msg:
        print("🛠️ วิธีแก้: ดาวน์โหลดไฟล์โมเดลไม่ได้ (เช็กอินเทอร์เน็ต)")
        print("   หรือโหลดเองแล้ววางไว้ในโฟลเดอร์ models:")
        print(f"   {HAND_MODEL_URL}")
        print(f"   {FACE_MODEL_URL}")
    elif "protobuf" in error_msg.lower() or "descriptor" in error_msg.lower():
        print("🛠️ วิธีแก้: เวอร์ชันของไลบรารี Protobuf ชนกัน")
        print("   python -m pip install --upgrade protobuf")
    else:
        print("🛠️ วิธีแก้เบื้องต้น: ลองอัปเดต mediapipe:")
        print("   python -m pip install --upgrade mediapipe")
    print("="*60 + "\n")
    sys.exit()

# ==========================================
# 2. ตั้งค่าพอร์ต, มุมกล้อง และสีไฟประจำอารมณ์
# ==========================================
COM_PORT = 'COM4'
BAUD_RATE = 115200

CAM_PAN_OFFSET = 0.0
CAM_TILT_OFFSET = 0.0

# ------------------------------------------
# กล้อง: 0 = กล้องในตัว laptop, 1 = กล้อง USB ตัวแรกที่เสียบเพิ่ม
# ถ้าเปิดไม่ได้ โปรแกรมจะลองเลขอื่นให้อัตโนมัติ (กล้อง USB ก่อน กล้อง laptop ทีหลัง)
# ------------------------------------------
CAMERA_INDEX = 1
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480

# ขนาดหน้าต่างกราฟตอนรัน: 1.0 = ขนาดเดิม (940 x 360), 1.4 = 1316 x 504 (พอดีจอ laptop 1366)
# จอใหญ่ 1920 ลอง 1.8 / ระหว่างรันกด + หรือ - เพื่อขยาย/ย่อได้เลย
GRAPH_SCALE = 1.4

# ทิศการหันตามคน (ภาพบนจอยังเป็นแบบกระจกเหมือนเดิม เปลี่ยนแค่ทิศที่แขนหัน)
PAN_INVERT  = True    # คนไปซ้ายแต่หุ่นหันขวา -> สลับ True/False
TILT_INVERT = False   # คนก้ม/ยืดตัวแล้วหุ่นเงย/ก้มผิดทาง -> สลับ True/False

LED_COLORS = {
    "STARTLED":  (180, 0, 255),
    "EXPLORING": (220, 220, 220),
    "HAPPY":     (255, 190, 0),
    "CURIOUS":   (0, 220, 60),
    "NORMAL":    (60, 200, 255),
    "ANGRY":     (255, 0, 0),
    "SAD":       (5, 10, 100),
    "SLEEPY":    (0, 0, 0),
    "QUIET":     (20, 20, 70),   # ไฟหรี่ ๆ ตอนถูกสั่งให้เงียบ
    "SULK":      (90, 0, 60)     # ม่วงแดงหรี่ ๆ ตอนงอน
}

ser = None
try:
    # สำคัญ: Maixduino ใช้ขา DTR/RTS สั่ง reset และเข้าโหมดโหลดเฟิร์มแวร์ (bootloader)
    # ถ้าเปิดพอร์ตแบบปกติ DTR/RTS จะกระตุกทันทีตอนเปิด -> บอร์ดค้างใน bootloader
    # (LED ไม่ติด มอเตอร์ไม่ขยับ) จึงต้องตั้ง DTR/RTS = False "ก่อน" เปิดพอร์ต
    ser = serial.Serial()
    ser.port = COM_PORT
    ser.baudrate = BAUD_RATE
    ser.timeout = 0.02
    ser.dtr = False
    ser.rts = False
    ser.open()
    print(f"⏳ กำลังรอให้บอร์ด Maixduino พร้อม (3 วินาที)...")
    time.sleep(3.0)
    if ser.in_waiting: ser.read(ser.in_waiting)
    print(f"✅ เชื่อมต่อ Maixduino ที่ {COM_PORT} สำเร็จ!")
except Exception as e:
    ser = None
    print(f"⚠️ รันโหมดจำลอง: เปิดพอร์ต {COM_PORT} ไม่ได้ ({e})")
    try:
        from serial.tools import list_ports
        ports = [f"{p.device} ({p.description})" for p in list_ports.comports()]
        print("   พอร์ตที่เจอในเครื่องตอนนี้:", ", ".join(ports) if ports else "ไม่มีเลย (เช็กสาย USB)")
    except Exception:
        pass
    print("   ถ้าเลขพอร์ตไม่ตรง ให้แก้ COM_PORT / ถ้าขึ้น 'Access is denied' ให้ปิด MaixPy IDE หรือ Serial Monitor ก่อน")

# ==========================================
# 3. จลนศาสตร์แขนกล 4 แกน
# ==========================================
L0, L1, L2 = 87.0, 92.0, 79.0
TH, TD = 50.0, 50.0
K_SERVO = 1023.0 / 200.0

ZERO = [405, 0, 0, 230]
DIR = [-1, 1, -1, 1]
EPROM_LIM = [(0, 810), (0, 650), (0, 600), (0, 450)]
ANGLE_LIM = [(-79.0, 79.0), (0.0, 127.0), (-117.0, 0.0), (-45.0, 43.0)]
HOME_Q = [0.0, 90.0, -90.0, 0.0]

# ------------------------------------------
# ท่าตอนเปิด / ปิดระบบ  (หน่วยองศา [ฐาน, ไหล่, ศอก, ข้อมือ] แก้ได้ตามชอบ)
# ------------------------------------------
STARTUP_Q  = [0.0, 90.0, -90.0, 0.0]      # ท่าพร้อมทำงานหลังเปิด
SHUTDOWN_Q = [0.0, 102.0, -102.0, 36.0]   # ท่าพักตอนปิด (ก้มหัวหลับ)
TORQUE_OFF_ON_EXIT = False                # True = ปล่อยแรงบิดมอเตอร์หลังปิด (ระวังแขนตก)

# ------------------------------------------
# ท่าชู่ว์ (มีคนเอานิ้วชี้แตะปาก)
# ------------------------------------------
QUIET_BACK_Q = [0.0, 100.0, -20.0, -15.0]   # ถอยหลัง ยืดหลังตรง (ฐานจะหันตามคนเอง ค่า 0 ตรงนี้ไม่ใช้)
QUIET_HUSH_PITCH = 8.0                      # ตอนเงียบ ก้มหัวลงนิด ๆ
QUIET_BACK_T = 0.8    # วินาที: ช่วงถอยหลัง
QUIET_NOD_T  = 1.6    # วินาที: ช่วงพยักหน้า
QUIET_HUSH_T = 4.0    # วินาที: ช่วงเงียบ (ชู่ว์ซ้ำ = เงียบต่อ)
NOD_COUNT, NOD_AMP = 2, 25.0   # พยักหน้า 2 ครั้ง ก้มลงครั้งละ 25 องศา
SHUSH_CONFIRM_FRAMES = 6       # ต้องเห็นท่าชู่ว์ต่อเนื่อง ~0.3 วิ ถึงจะเชื่อ (กันมั่ว)

# ------------------------------------------
# หลอดอารมณ์ (mood): บวกเยอะ = ขยับเร็ว ลื่นไหล / ลบเยอะ = ขยับช้า เนือย ๆ
# ------------------------------------------
SPEED_MAX_MULT = 1.5    # mood = +1 -> เร็วขึ้น 1.5 เท่า
SPEED_MIN_MULT = 0.5    # mood = -1 -> ช้าลงเหลือครึ่งเดียว
MOOD_RATE      = 0.4    # mood เปลี่ยนเร็วแค่ไหน (น้อย = สะสมช้า เปลี่ยนนิสัยช้า)

# ------------------------------------------
# หลอดงอน (แยกจากความโกรธปกติ) เต็มแล้วจะหันหน้าหนี
# ------------------------------------------
SULK_FILL_RATE      = 0.20   # เติมต่อวินาทีขณะโกรธเต็มที่ (โกรธสะสมรวม ~5 วิ = เต็มหลอด)
SULK_DECAY_RATE     = 0.01   # ตอนยังไม่งอน หลอดค่อย ๆ ลดเองช้ามาก
SULK_SELF_HEAL_RATE = 0.035  # ตอนงอนอยู่ หายเองได้ (~20 วิ ถ้าไม่มีใครง้อ)
SULK_WAVE_HEAL_RATE = 0.25   # โบกมือง้อ ลดหลอดเร็ว (~3 วิ)
SULK_EXIT_LEVEL     = 0.30   # หลอดลดถึงระดับนี้ = หายงอน

def clamp(val, lo, hi): return max(lo, min(hi, val))

def deg_to_servo(qd):
    return [int(clamp(round(ZERO[i] + DIR[i] * qd[i] * K_SERVO), EPROM_LIM[i][0], EPROM_LIM[i][1])) for i in range(4)]

def fk(qd):
    q1, q2, q3, q4 = [math.radians(v) for v in qd]
    a2, a3, a4 = q2, q2 + q3, q2 + q3 + q4
    r_el, z_el = L1 * math.cos(a2), L0 + L1 * math.sin(a2)
    r_wr, z_wr = r_el + L2 * math.cos(a3), z_el + L2 * math.sin(a3)
    r_nk, z_nk = r_wr - TH * math.sin(a4), z_wr + TH * math.cos(a4)
    r_tcp, z_tcp = r_nk + TD * math.cos(a4), z_nk + TD * math.sin(a4)
    pts = [(0.0, 0.0), (0.0, L0), (r_el, z_el), (r_wr, z_wr), (r_nk, z_nk), (r_tcp, z_tcp)]
    return {"r": r_tcp, "pts": pts}

def is_safe(qd):
    for i in range(4):
        if qd[i] < ANGLE_LIM[i][0] - 1e-4 or qd[i] > ANGLE_LIM[i][1] + 1e-4: return False
    res = fk(qd)
    pts = res["pts"]
    for k in range(2, len(pts)):
        if pts[k][1] < 20.0: return False
    for k in range(1, len(pts) - 1):
        for n in range(1, 9):
            a = n / 8.0
            r = pts[k][0] + (pts[k+1][0] - pts[k][0]) * a
            z = pts[k][1] + (pts[k+1][1] - pts[k][1]) * a
            if abs(r) < 45.0 and z < 75.0: return False
    s_idx = abs(math.sin(math.radians(qd[2]))) * min(1.0, abs(res["r"]) / 48.1)
    return s_idx >= 0.1

def solve_ik(x, y, z, phi_pref, prev_q1=0.0):
    r = math.hypot(x, y)
    q1 = math.radians(prev_q1) if r < 1e-6 else math.atan2(y, x)
    for d in range(0, 90):
        for p_deg in ([phi_pref] if d == 0 else [phi_pref + d, phi_pref - d]):
            phi = math.radians(p_deg)
            rw = r - TD * math.cos(phi) + TH * math.sin(phi)
            zw = z - L0 - TD * math.sin(phi) - TH * math.cos(phi)
            D = (rw*rw + zw*zw - L1*L1 - L2*L2) / (2.0 * L1 * L2)
            if abs(D) <= 1.0:
                q3 = -math.acos(D)
                q2 = math.atan2(zw, rw) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
                q4 = phi - q2 - q3
                qd = [math.degrees(q1), math.degrees(q2), math.degrees(q3), math.degrees(q4)]
                if is_safe(qd): return qd
    return None

# ==========================================
# 4. โมเดล Bio-Inspired (Hormone & CPG)
# ==========================================
class ArtificialHormone:
    def __init__(self, k_prod=0.8, k_clear=0.6, k_bind=0.2, r_gain=1.0, fast_clear_mult=1.0):
        self.k_prod, self.k_clear, self.k_bind = k_prod, k_clear, k_bind
        self.r_gain, self.fast_clear_mult = r_gain, fast_clear_mult
        self.H = self.R = 0.0

    def step(self, S, dt=0.05):
        eff_S = max(0.0, S)
        eff_clear = self.k_clear if S >= 0 else self.k_clear * self.fast_clear_mult
        P = self.k_prod * eff_S
        C = eff_clear * self.H
        B = self.k_bind * self.R
        self.H = clamp(self.H + dt * (P - C - B), 0.0, 1.0)
        self.R = clamp(self.r_gain * self.H, 0.0, 1.0)
        return self.R

class SO2_CPG:
    def __init__(self):
        self.a1, self.a2 = 0.1, 0.0
        self.o1, self.o2 = math.tanh(self.a1), math.tanh(self.a2)

    def step(self, alpha_cmd=1.1, phi_cmd=0.2):
        w11 = w22 = alpha_cmd * math.cos(phi_cmd)
        w12 = alpha_cmd * math.sin(phi_cmd)
        w21 = -alpha_cmd * math.sin(phi_cmd)
        o1_old, o2_old = self.o1, self.o2
        self.a1 = w11 * o1_old + w12 * o2_old
        self.a2 = w21 * o1_old + w22 * o2_old
        self.o1, self.o2 = math.tanh(self.a1), math.tanh(self.a2)
        return self.o1, self.o2

# ==========================================
# 5. วิเคราะห์ด้วย AI (MediaPipe Tasks API)
# ==========================================
def draw_hand(frame, lms):
    h_img, w_img = frame.shape[:2]
    pts = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1, cv2.LINE_AA)

def analyze_frame_ai(frame, rgb_frame, timestamp_ms):
    h_img, w_img = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # ข้อมูลเสริมสำหรับท่าชู่ว์
    mouth, face_w_norm = None, 0.0
    index_tip, index_only = None, False

    # 1. AI จับใบหน้า
    face_rect = None
    face_res = face_detector.detect_for_video(mp_image, timestamp_ms)
    if face_res.detections:
        detection = max(face_res.detections,
                        key=lambda d: d.categories[0].score if d.categories else 0.0)
        bbox = detection.bounding_box  # หน่วยเป็นพิกเซลแล้ว
        x, y = int(bbox.origin_x), int(bbox.origin_y)
        w, h = int(bbox.width), int(bbox.height)
        face_rect = (max(0, x), max(0, y), w, h)
        cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[0] + w, face_rect[1] + h), (0, 255, 128), 2)
        cv2.putText(frame, "Human Face", (face_rect[0], max(15, face_rect[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

        # หาตำแหน่งปาก: FaceDetector ให้จุดสำคัญ 6 จุด จุดที่ 3 คือปาก
        kps = getattr(detection, "keypoints", None) or []
        if len(kps) > 3:
            mouth = (clamp(kps[3].x, 0.0, 1.0), clamp(kps[3].y, 0.0, 1.0))
        else:
            mouth = ((x + w / 2.0) / w_img, (y + 0.78 * h) / h_img)   # เดาจากกรอบหน้า
        face_w_norm = w / float(w_img)
        cv2.circle(frame, (int(mouth[0] * w_img), int(mouth[1] * h_img)), 4, (255, 0, 255), -1)

    # 2. AI จับมือและท่าทาง
    gesture = "NONE"
    fingers = 0
    hand_pos_norm = None
    hand_size_norm = 0.0

    hand_res = hands_detector.detect_for_video(mp_image, timestamp_ms)
    if hand_res.hand_landmarks:
        lms = hand_res.hand_landmarks[0]
        draw_hand(frame, lms)

        xs, ys = [lm.x for lm in lms], [lm.y for lm in lms]
        hand_size_norm = clamp((max(xs) - min(xs)) * (max(ys) - min(ys)), 0.0, 1.0)

        tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            if lms[tip].y < lms[pip].y: fingers += 1
        if abs(lms[4].x - lms[5].x) > 0.06: fingers += 1

        hx, hy = clamp(lms[9].x, 0.0, 1.0), clamp(lms[9].y, 0.0, 1.0)
        hand_pos_norm = (hx, hy)

        # นิ้วชี้ตั้งนิ้วเดียว (นิ้วกลาง นาง ก้อย พับ / ไม่สนนิ้วโป้ง)
        index_tip = (lms[8].x, lms[8].y)
        index_only = (lms[8].y < lms[6].y and lms[12].y > lms[10].y
                      and lms[16].y > lms[14].y and lms[20].y > lms[18].y)

        if fingers == 0:
            gesture, col = 'FIST', (0, 0, 255)
        elif fingers <= 2:
            gesture, col = 'POINT_PEACE', (0, 220, 60)
        else:
            gesture, col = 'OPEN_PALM', (0, 200, 255)

        px, py = int(hx * w_img), int(hy * h_img)
        cv2.putText(frame, f"{gesture} ({fingers}F)", (max(10, px - 40), max(25, py - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    extra = {"mouth": mouth, "face_w": face_w_norm, "index_tip": index_tip, "index_only": index_only}
    return face_rect, gesture, hand_pos_norm, hand_size_norm, extra

# ==========================================
# 6. ระบบส่งคำสั่งมอเตอร์ (MoveJ)
# ==========================================
current_q = list(HOME_Q)
led_rgb = LED_COLORS["EXPLORING"]
estop_active = False

# ------------------------------------------
# ระบบลดอาการสั่น
# ------------------------------------------
SMOOTH_GAIN     = 1.0     # มาก = ตอบสนองไวขึ้นแต่แข็งขึ้น / น้อย = นุ่มขึ้นแต่ช้าลง (ลอง 0.7 - 1.3)
DEADBAND_COUNTS = 2       # เป้าเปลี่ยนน้อยกว่ากี่หน่วยเซอร์โว (1 หน่วย ~0.2 องศา) ไม่ต้องส่ง -> กันมอเตอร์ส่ายไปมา
TARGET_DEADZONE = 0.012   # คน/มือขยับน้อยกว่า 1.2% ของจอ ถือว่าไม่ขยับ -> กันสัญญาณ AI สั่น
SWAY_SCALE      = 1.0     # ความแรงการแกว่งตัว (CPG) ทุกท่า: 1.0 = เท่าเดิม, 0.6 = นิ่งขึ้น

motion_target_q = list(HOME_Q)   # เป้าหมายล่าสุดจากลูปหลัก
motion_stage1   = list(HOME_Q)   # ตัวกรองชั้นที่ 1 (ชั้นที่ 2 คือ current_q)
motion_tc       = 0.2            # time constant ของตัวกรอง (วินาที)
servo_spd_cmd   = 25

def send_serial(pos_list, spd_ms, rgb, tq=1):
    if ser and ser.is_open and not estop_active:
        cmd = f"{pos_list[0]},{pos_list[1]},{pos_list[2]},{pos_list[3]},{spd_ms},{rgb[0]},{rgb[1]},{rgb[2]},{tq}\r\n"
        try: ser.write(cmd.encode('utf-8')); ser.flush()
        except: pass

speed_mult = 1.0   # ตัวคูณความเร็วจาก mood (0.5 - 1.5)

def duration_to_tc(duration):
    """แปลง duration แบบเดิมเป็น time constant ของตัวกรอง ให้ความไวใกล้ของเดิม"""
    s = min(1.0, 0.05 / max(0.08, duration))
    frac = (s * s) * (3.0 - 2.0 * s)            # ของเดิมขยับรอบละกี่ % ของระยะที่เหลือ
    T = -0.05 / math.log(max(1e-6, 1.0 - frac))  # time constant ที่เทียบเท่า
    return max(0.02, 0.5 * T / SMOOTH_GAIN)      # กรอง 2 ชั้น -> แต่ละชั้นใช้ครึ่งหนึ่ง

def set_auto_movej(target_q, duration=0.35, spd=25, scale=True):
    """แค่ 'ตั้งเป้า' ให้ thread มอเตอร์ไล่ตามแบบนุ่ม ๆ (ไม่รีเซ็ตการเคลื่อนที่ทุกเฟรมเหมือนเดิม)"""
    global motion_target_q, motion_tc, servo_spd_cmd
    if target_q is None: return
    if scale:
        # mood ดี -> duration สั้นลง (ไล่ตามเป้าไวและลื่นขึ้น) + มอเตอร์เร็วขึ้น / mood แย่ -> กลับกัน
        duration = duration / speed_mult
        spd = spd * speed_mult
    servo_spd_cmd = int(clamp(spd, 5, 100))   # ห้ามเป็น 0 เพราะเซอร์โวตีความ 0 = เร็วสุด
    motion_tc = duration_to_tc(duration)
    motion_target_q = [clamp(target_q[i], ANGLE_LIM[i][0], ANGLE_LIM[i][1]) for i in range(4)]

def move_blocking(target_q, duration, spd, rgb_from, rgb_to):
    """ขยับแขนแบบรอจนเสร็จ + ไล่สีไฟ ใช้ตอนเปิด/ปิดเท่านั้น (ตอนที่ thread มอเตอร์ไม่ได้ทำงาน)"""
    global current_q, motion_target_q, motion_stage1
    start = list(current_q)
    steps = max(1, int(duration / 0.05))
    for k in range(1, steps + 1):
        s = k / steps
        e = (s * s) * (3.0 - 2.0 * s)
        current_q = [start[i] + (target_q[i] - start[i]) * e for i in range(4)]
        rgb = tuple(int(rgb_from[j] + (rgb_to[j] - rgb_from[j]) * s) for j in range(3))
        send_serial(deg_to_servo(current_q), spd, rgb)
        time.sleep(0.05)
    motion_target_q, motion_stage1 = list(current_q), list(current_q)

def startup_sequence():
    """ตอนเปิด: ไม่รู้ว่าแขนค้างอยู่ท่าไหน -> ค่อย ๆ ไปท่าพักก่อน แล้วค่อย ๆ 'ตื่น' ขึ้นมาท่าพร้อมทำงาน"""
    global current_q
    for name, q in [("STARTUP_Q", STARTUP_Q), ("SHUTDOWN_Q", SHUTDOWN_Q)]:
        if not is_safe(q): print(f"⚠️ ท่า {name} = {q} อาจชนตัวเอง/เกินลิมิต ลองแก้ตัวเลขดูนะ")
    print("🌅 กำลังตื่นนอน...")
    current_q = list(SHUTDOWN_Q)
    send_serial(deg_to_servo(SHUTDOWN_Q), 15, (0, 0, 0))
    time.sleep(2.0)
    move_blocking(STARTUP_Q, 1.5, 20, (0, 0, 0), LED_COLORS["EXPLORING"])
    print("✅ พร้อมทำงานแล้ว!")

def shutdown_sequence():
    """ตอนปิด: ค่อย ๆ ก้มหัวลงท่าพัก ไฟหรี่จนดับ"""
    print("🌙 กำลังเข้านอน...")
    move_blocking(SHUTDOWN_Q, 2.0, 15, led_rgb, (0, 0, 0))
    time.sleep(0.5)
    if TORQUE_OFF_ON_EXIT:
        send_serial(deg_to_servo(SHUTDOWN_Q), 15, (0, 0, 0), tq=0)

def motion_control_thread():
    """ไล่ตามเป้าด้วยตัวกรอง 2 ชั้น (เร่ง-ผ่อนนุ่ม ๆ ไม่กระตุก) ด้วยจังหวะเวลาคงที่ 20 ครั้ง/วินาที"""
    global current_q, motion_stage1
    dt = 0.05
    last_cmd, last_rgb, last_send = None, None, 0.0
    next_t = time.perf_counter()
    while not stop_threads:
        # จังหวะคงที่: ไม่สะสมความคลาดเคลื่อนของ sleep
        next_t += dt
        rem = next_t - time.perf_counter()
        if rem > 0: time.sleep(rem)
        else: next_t = time.perf_counter()   # ช้ากว่ากำหนด -> เริ่มนับใหม่ ไม่เร่งชดเชย
        if estop_active: continue

        a = 1.0 - math.exp(-dt / motion_tc)
        tgt = motion_target_q
        motion_stage1 = [motion_stage1[i] + a * (tgt[i] - motion_stage1[i]) for i in range(4)]
        current_q = [current_q[i] + a * (motion_stage1[i] - current_q[i]) for i in range(4)]

        cmd = deg_to_servo(current_q)
        rgb = tuple(led_rgb)
        now = time.perf_counter()
        moved = last_cmd is None or max(abs(cmd[i] - last_cmd[i]) for i in range(4)) >= DEADBAND_COUNTS
        if moved or rgb != last_rgb or now - last_send > 0.5:   # ส่งซ้ำทุก 0.5 วิ กันคำสั่งหาย
            send_serial(cmd, servo_spd_cmd, rgb)
            last_cmd, last_rgb, last_send = cmd, rgb, now

# ==========================================
# 7. วาดกราฟ Real-Time
# ==========================================
FONT = cv2.FONT_HERSHEY_SIMPLEX

def _s(v):
    """แปลงพิกัด/ขนาดจากหน่วยออกแบบ (ขนาดเดิม) เป็นพิกเซลจริงตาม GRAPH_SCALE"""
    return int(round(v * GRAPH_SCALE))

def _txt(img, text, x, y, fs, col, th=1):
    cv2.putText(img, text, (_s(x), _s(y)), FONT, fs * GRAPH_SCALE, col, max(1, round(th * GRAPH_SCALE)), cv2.LINE_AA)

def _line(img, p1, p2, col, th=1):
    cv2.line(img, (_s(p1[0]), _s(p1[1])), (_s(p2[0]), _s(p2[1])), col, max(1, round(th * GRAPH_SCALE)), cv2.LINE_AA)

def _rect(img, p1, p2, col, th=1):
    t = -1 if th < 0 else max(1, round(th * GRAPH_SCALE))
    cv2.rectangle(img, (_s(p1[0]), _s(p1[1])), (_s(p2[0]), _s(p2[1])), col, t)

def draw_bars(panel, mood, sulk, speed):
    """หลอด Mood (ลบ <- กลาง -> บวก) และหลอด Sulk (งอน) ใต้กราฟ (หน่วยออกแบบ: แผง 460 x 360)"""
    w, h = 460, 360
    bx0, bx1 = 70, w - 62
    bw = bx1 - bx0
    # หลอด Mood
    y0, y1 = h - 33, h - 23
    _rect(panel, (bx0, y0), (bx1, y1), (55, 55, 65), -1)
    mid = bx0 + bw / 2
    end = mid + clamp(mood, -1.0, 1.0) * (bw / 2)
    col = (80, 220, 80) if mood >= 0 else (200, 120, 60)
    _rect(panel, (min(mid, end), y0), (max(mid, end), y1), col, -1)
    _line(panel, (mid, y0 - 2), (mid, y1 + 2), (200, 200, 200))
    _txt(panel, "Mood", 8, y1, 0.40, (200, 200, 200))
    _txt(panel, f"x{speed:.2f}", bx1 + 8, y1, 0.40, (200, 200, 200))
    # หลอดงอน
    y0, y1 = h - 17, h - 7
    _rect(panel, (bx0, y0), (bx1, y1), (55, 55, 65), -1)
    _rect(panel, (bx0, y0), (bx0 + clamp(sulk, 0.0, 1.0) * bw, y1), (160, 60, 220), -1)
    xe = bx0 + SULK_EXIT_LEVEL * bw
    _line(panel, (xe, y0 - 2), (xe, y1 + 2), (200, 200, 200))
    _txt(panel, "Sulk", 8, y1, 0.40, (200, 200, 200))

def draw_hormone_graph(hist_att, hist_ang, hist_str, hist_rst, state_mode, gesture_name, phi_cmd,
                       mood=0.0, sulk=0.0, speed=1.0):
    w, h = 460, 360   # หน่วยออกแบบ (ขนาดจริง = x GRAPH_SCALE)
    panel = np.zeros((_s(h), _s(w), 3), dtype=np.uint8)
    panel[:] = (28, 28, 34)

    gx0, gy0, gx1, gy1 = 48, 72, w - 20, h - 38
    gw, gh = gx1 - gx0, gy1 - gy0
    _rect(panel, (gx0, gy0), (gx1, gy1), (42, 42, 50), -1)
    _rect(panel, (gx0, gy0), (gx1, gy1), (90, 90, 105), 1)

    for val in [0.0, 0.25, 0.50, 0.75, 1.0]:
        y_px = gy1 - val * gh
        _line(panel, (gx0, y_px), (gx1, y_px), (60, 60, 72))
        _txt(panel, f"{val:.2f}", 6, y_px + 4, 0.37, (180, 180, 190))

    y_slp = gy1 - 0.90 * gh
    _line(panel, (gx0, y_slp), (gx1, y_slp), (150, 150, 150))
    _txt(panel, "SLEEP (0.90)", gx0 + 6, y_slp - 4, 0.33, (175, 175, 175))

    y_happy = gy1 - 0.75 * gh
    _line(panel, (gx0, y_happy), (gx1, y_happy), (0, 200, 255))
    _txt(panel, "HAPPY/NORM (0.75)", gx1 - 125, y_happy - 4, 0.33, (0, 200, 255))

    y_ang = gy1 - 0.55 * gh
    _line(panel, (gx0, y_ang), (gx1, y_ang), (70, 70, 255))
    _txt(panel, "ANGRY / SAD (0.55)", gx0 + 6, y_ang - 4, 0.33, (100, 100, 255))

    def plot_curve(data_deque, color):
        if len(data_deque) < 2: return
        pts = [(_s(gx0 + (i / (len(data_deque) - 1)) * gw), _s(gy1 - clamp(v, 0.0, 1.0) * gh)) for i, v in enumerate(data_deque)]
        cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, color, max(2, round(2 * GRAPH_SCALE)), cv2.LINE_AA)

    plot_curve(hist_rst, (255, 190, 70))
    plot_curve(hist_str, (255, 50, 200))
    plot_curve(hist_ang, (40, 40, 255))
    plot_curve(hist_att, (0, 210, 255))

    if state_mode.startswith("QUIET"): badge_col = (255, 180, 255)
    elif state_mode == "SULK":         badge_col = (200, 80, 220)
    else:                              badge_col = LED_COLORS.get(state_mode, (200, 200, 200))
    _txt(panel, f"MODE: {state_mode}", 15, 26, 0.62, badge_col, 2)
    _txt(panel, f"Hand: {gesture_name}", 265, 25, 0.46, (255, 220, 120))

    c_att, c_ang = hist_att[-1] if hist_att else 0, hist_ang[-1] if hist_ang else 0
    c_rst, c_str = hist_rst[-1] if hist_rst else 0, hist_str[-1] if hist_str else 0
    _txt(panel, f"Happy:{c_att:.2f}", 15, 52, 0.40, (0, 210, 255))
    _txt(panel, f"Angry:{c_ang:.2f}", 120, 52, 0.40, (50, 50, 255))
    _txt(panel, f"Sad/Slp:{c_rst:.2f}", 225, 52, 0.40, (255, 190, 70))
    _txt(panel, f"Startle:{c_str:.2f}", 345, 52, 0.40, (255, 50, 200))
    draw_bars(panel, mood, sulk, speed)
    return panel

# ==========================================
# 8. ลูปหลัก
# ==========================================
stop_threads = False

WIN_NAME = "Auto-Bot Bio-Vision & Real-Time Emotion Graph"

def set_graph_scale(v):
    global GRAPH_SCALE
    GRAPH_SCALE = round(clamp(v, 0.8, 2.5), 1)
    try: cv2.resizeWindow(WIN_NAME, _s(940), _s(360))
    except Exception: pass
    print(f"🔍 ขนาดกราฟ x{GRAPH_SCALE}")

def try_open_camera(index):
    """เปิดกล้องตามเลข แล้วลองอ่านภาพจริง 1 เฟรม (บางทีเปิดได้แต่ไม่มีภาพ)"""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # Windows: เปิดเร็วกว่า และใช้กับกล้อง USB ได้ดีกว่า
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # ลดภาระสาย USB
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    for _ in range(10):                       # กล้อง USB บางตัวต้องอุ่นเครื่องสักแป๊บ
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
        time.sleep(0.05)
    cap.release()
    return None

def open_camera():
    order = [CAMERA_INDEX] + [i for i in (1, 2, 3, 0) if i != CAMERA_INDEX]
    for idx in order:
        cap = try_open_camera(idx)
        if cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = "กล้องในตัว laptop (น่าจะ)" if idx == 0 else "กล้อง USB (น่าจะ)"
            if idx != CAMERA_INDEX:
                print(f"⚠️ เปิดกล้องเลข {CAMERA_INDEX} ไม่ได้ ใช้กล้องเลข {idx} แทน")
            print(f"📷 ใช้กล้องเลข {idx} - {name} ({w}x{h})")
            return cap
    print("❌ ไม่พบกล้องที่ใช้งานได้เลย: เช็กสาย USB / ปิดโปรแกรมอื่นที่ใช้กล้องอยู่ (Zoom, Teams, แอปกล้อง)")
    return None

def main_loop(cap):
    global led_rgb, speed_mult

    h_attentive = ArtificialHormone(k_prod=0.35, k_clear=0.20, k_bind=0.05)
    # k_clear 0.20 -> 0.30 : ฮอร์โมนโกรธสลายเร็วขึ้น (หายโกรธไวขึ้น)
    h_anger     = ArtificialHormone(k_prod=0.60, k_clear=0.30, k_bind=0.05, fast_clear_mult=4.0)
    h_startle   = ArtificialHormone(k_prod=6.5, k_clear=1.5, k_bind=0.2)
    h_rest      = ArtificialHormone(k_prod=0.14, k_clear=0.08, k_bind=0.0, fast_clear_mult=2.5)
    cpg = SO2_CPG()

    MAX_PTS = 200
    hist_att, hist_ang = deque([0.0]*MAX_PTS, maxlen=MAX_PTS), deque([0.0]*MAX_PTS, maxlen=MAX_PTS)
    hist_str, hist_rst = deque([0.0]*MAX_PTS, maxlen=MAX_PTS), deque([0.0]*MAX_PTS, maxlen=MAX_PTS)

    beta_filter = 0.10
    cx_filt, cy_filt, prev_cx, prev_cy = 0.5, 0.5, 0.5, 0.5
    prev_size, prev_hx, prev_hy, prev_hsize = 0.0, 0.5, 0.5, 0.0
    state_mode = "EXPLORING"

    # ระบบจับท่าโบกมือ (เก็บตำแหน่งมือแนวนอนย้อนหลัง ~1.5 วินาที)
    wave_hist = deque(maxlen=60)
    wave_level = 0.0   # 0 = ไม่โบก, 1 = โบกเต็มที่ (ค่อย ๆ จางหายเมื่อหยุดโบก)

    # ระบบท่าชู่ว์ (นิ้วชี้แตะปาก)
    quiet_phase, quiet_t0, quiet_yaw = None, 0.0, 0.0   # quiet_phase: None / BACK / NOD / HUSH
    shush_frames = 0
    last_mouth, last_mouth_t, last_face_w = None, 0.0, 0.15

    # หลอดอารมณ์ + หลอดงอน
    mood = 0.0          # -1 (ลบ) ... +1 (บวก)
    sulk = 0.0          # 0 ... 1 (เต็ม = งอน)
    sulking, sulk_side = False, 1

    t_start = time.time()
    last_ts_ms = -1
    fail_count = 0
    dt = 0.05

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            # กล้อง USB หลุด / สายหลวม: รอสักพัก ถ้าเกิน ~3 วิ ให้ปิดโปรแกรม (เข้าท่าปิดให้อัตโนมัติ)
            fail_count += 1
            if fail_count == 1: print("⚠️ อ่านภาพจากกล้องไม่ได้ กำลังรอ...")
            if fail_count > 60:
                print("❌ กล้องหลุด! ปิดระบบ")
                break
            cv2.waitKey(1); time.sleep(0.05)
            continue
        fail_count = 0

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (480, 360))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # โหมด VIDEO ต้องส่ง timestamp (ms) ที่เพิ่มขึ้นเรื่อย ๆ ทุกเฟรม
        ts_ms = int((time.time() - t_start) * 1000)
        if ts_ms <= last_ts_ms: ts_ms = last_ts_ms + 1
        last_ts_ms = ts_ms

        # วิเคราะห์เฟรมด้วย AI (MediaPipe) ล้วนๆ
        face_rect, gesture_name, hand_pos, hand_size, extra = analyze_frame_ai(frame, rgb_frame, ts_ms)

        S_face, S_spike, S_anger = 0.0, 0.0, 0.0
        target_x, target_y = cx_filt, cy_filt

        # 0. ตรวจจับท่าโบกมือสวัสดี: แบมือ + มือส่ายซ้าย-ขวากลับไปกลับมา
        now = time.time()
        if hand_pos is not None and gesture_name == "OPEN_PALM":
            wave_hist.append((now, hand_pos[0]))
        else:
            wave_hist.clear()
        while wave_hist and now - wave_hist[0][0] > 1.5:
            wave_hist.popleft()

        is_waving = False
        if len(wave_hist) >= 6:
            xs_w = [p[1] for p in wave_hist]
            flips, last_dir, anchor = 0, 0, xs_w[0]
            for xw in xs_w[1:]:
                d = xw - anchor
                if abs(d) >= 0.04:              # ต้องขยับเกิน 4% ของจอถึงนับว่าเปลี่ยนทิศ (กันมือสั่น)
                    dir_now = 1 if d > 0 else -1
                    if last_dir != 0 and dir_now != last_dir:
                        flips += 1
                    last_dir, anchor = dir_now, xw
            if flips >= 2 and (max(xs_w) - min(xs_w)) >= 0.08:
                is_waving = True

        wave_level = min(1.0, wave_level + 0.15) if is_waving else max(0.0, wave_level - 0.03)
        if wave_level > 0.3:
            cv2.putText(frame, "WAVE! Hello!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 0.5 ตรวจจับท่าชู่ว์: นิ้วชี้ตั้งนิ้วเดียว + ปลายนิ้วอยู่ใกล้ปาก
        #     (นิ้วบังปากแล้ว AI อาจจับหน้าหลุด -> จำตำแหน่งปากล่าสุดไว้ 1 วินาที)
        if extra["mouth"] is not None:
            last_mouth, last_mouth_t, last_face_w = extra["mouth"], now, max(0.05, extra["face_w"])
        mouth_ref = last_mouth if (last_mouth is not None and now - last_mouth_t < 1.0) else None

        shush_now = False
        if extra["index_only"] and extra["index_tip"] is not None and mouth_ref is not None:
            d_mouth = math.hypot(extra["index_tip"][0] - mouth_ref[0], extra["index_tip"][1] - mouth_ref[1])
            if d_mouth < max(0.06, 0.45 * last_face_w):
                shush_now = True
        shush_frames = shush_frames + 1 if shush_now else 0
        if shush_now:
            gesture_name = "SHUSH"
            cv2.putText(frame, "Shhh...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 255), 2)

        if shush_frames >= SHUSH_CONFIRM_FRAMES:
            if quiet_phase is None:
                quiet_t0 = now
                quiet_yaw = clamp((0.5 - cx_filt + CAM_PAN_OFFSET) * 110.0, -70.0, 70.0)   # หันหาคนที่สั่ง
            elif quiet_phase == "HUSH":
                quiet_t0 = now - (QUIET_BACK_T + QUIET_NOD_T)   # ชู่ว์ซ้ำตอนเงียบอยู่ = เงียบต่อ

        # ตอนนี้อยู่ช่วงไหนของท่าชู่ว์
        if quiet_t0 > 0.0:
            el_q = now - quiet_t0
            if el_q < QUIET_BACK_T: quiet_phase = "BACK"
            elif el_q < QUIET_BACK_T + QUIET_NOD_T: quiet_phase = "NOD"
            elif el_q < QUIET_BACK_T + QUIET_NOD_T + QUIET_HUSH_T: quiet_phase = "HUSH"
            else: quiet_phase, quiet_t0 = None, 0.0

        # 1. จัดการข้อมูลใบหน้า
        if face_rect is not None:
            S_face = 1.0
            x, y, w, h = face_rect
            face_x, face_y = (x + w / 2.0) / 480.0, (y + h / 2.0) / 360.0
            target_x = 1.0 - face_x if PAN_INVERT else face_x
            target_y = 1.0 - face_y if TILT_INVERT else face_y
            size_filt = beta_filter * ((w * h) / (480.0 * 360.0)) + (1.0 - beta_filter) * prev_size

            if (size_filt - prev_size > 0.035) or (math.hypot(face_x - prev_cx, face_y - prev_cy) > 0.12):
                S_spike = 1.0
            prev_cx, prev_cy, prev_size = face_x, face_y, size_filt

        # 2. จัดการข้อมูลมือ
        if hand_pos is not None:
            hx, hy = hand_pos
            target_x = 1.0 - hx if PAN_INVERT else hx
            target_y = 1.0 - hy if TILT_INVERT else hy
            S_face = max(S_face, 1.0)
            hand_vel = math.hypot(hx - prev_hx, hy - prev_hy)
            hand_loom = hand_size - prev_hsize

            # ตอนกำลังโบกมือ ไม่ให้ตกใจจากมือที่ขยับเร็ว
            if wave_level < 0.3 and (hand_loom > 0.060 or hand_vel > 0.20 or (prev_hsize == 0.0 and hand_size > 0.15)):
                S_spike = 1.0
            prev_hx, prev_hy, prev_hsize = hx, hy, hand_size
        else:
            prev_hsize = 0.0

        if gesture_name == "FIST":
            if S_spike == 0.0: S_anger = 1.0
        elif gesture_name == "OPEN_PALM":
            S_anger, S_face = -1.0, 1.4
        if is_waving:
            S_face = 1.8   # โบกมือ = ดีใจขึ้นเร็วกว่าแบมือเฉย ๆ
        if quiet_phase is not None or shush_now:
            S_spike, S_anger = 0.0, -1.0   # ตอนถูกสั่งให้เงียบ: ไม่ตกใจ ไม่โกรธ ใจเย็นลง

        # deadzone: ขยับนิดเดียว (สัญญาณ AI สั่น) ไม่ต้องตาม
        if abs(target_x - cx_filt) < TARGET_DEADZONE: target_x = cx_filt
        if abs(target_y - cy_filt) < TARGET_DEADZONE: target_y = cy_filt
        cx_filt = beta_filter * target_x + (1.0 - beta_filter) * cx_filt
        cy_filt = beta_filter * target_y + (1.0 - beta_filter) * cy_filt

        # 3. อัปเดตฮอร์โมน
        R_att = h_attentive.step(S_face, dt)
        R_ang = h_anger.step(S_anger, dt)
        R_str = h_startle.step(S_spike, dt)

        if (S_face > 0) or (gesture_name != "NONE") or (state_mode == "SLEEPY"):
            R_rst = h_rest.step(-1.0, dt)
        else:
            R_rst = h_rest.step(1.0, dt)

        hist_att.append(R_att); hist_ang.append(R_ang)
        hist_str.append(R_str); hist_rst.append(R_rst)

        # อารมณ์บวก (ดีใจ, ถูกทักทาย) vs ลบ (เหงา/ง่วง, ตกใจ) สะสมเป็น mood แล้วคุมความเร็ว
        mood_pos = R_att + 0.5 * wave_level
        mood_neg = R_rst + 0.6 * R_str
        mood_target = clamp(mood_pos - mood_neg, -1.0, 1.0)
        mood += (mood_target - mood) * clamp(MOOD_RATE * dt, 0.0, 1.0)
        if mood >= 0: speed_mult = 1.0 + (SPEED_MAX_MULT - 1.0) * mood
        else:         speed_mult = 1.0 + (1.0 - SPEED_MIN_MULT) * mood

        # หลอดงอน: โกรธ = เติม / โบกมือง้อ = ลดเร็ว / ปล่อยไว้ = หายเองช้า ๆ
        sulk += dt * SULK_FILL_RATE * R_ang
        if wave_level > 0.3:
            sulk -= dt * SULK_WAVE_HEAL_RATE * wave_level
        sulk -= dt * (SULK_SELF_HEAL_RATE if sulking else SULK_DECAY_RATE)
        sulk = clamp(sulk, 0.0, 1.0)
        if not sulking and sulk >= 1.0:
            sulking = True
            person_yaw0 = (0.5 - cx_filt + CAM_PAN_OFFSET) * 110.0
            sulk_side = -1 if person_yaw0 >= 0 else 1    # หันไปฝั่งตรงข้ามกับคน
        elif sulking and sulk <= SULK_EXIT_LEVEL:
            sulking = False

        # โบกมือ -> CPG แกว่งถี่ขึ้น / mood ดี -> จังหวะคึกขึ้น
        phi_cmd = clamp(0.20 + 0.20 * R_att + 0.25 * R_ang - 0.10 * R_rst + 0.12 * wave_level + 0.06 * mood, 0.10, 0.50)
        o1, o2 = cpg.step(phi_cmd=phi_cmd)
        o1, o2 = o1 * SWAY_SCALE, o2 * SWAY_SCALE

        # 4. State Machine ถ่างเกณฑ์ให้เปลี่ยนอารมณ์ยากขึ้น (นิ่งขึ้น)
        if R_str >= 0.55: state_mode = "STARTLED"
        elif state_mode == "STARTLED" and R_str >= 0.25: state_mode = "STARTLED"
        elif R_ang >= 0.55: state_mode = "ANGRY"
        elif state_mode == "ANGRY" and R_ang >= 0.32: state_mode = "ANGRY"   # 0.25 -> 0.32 ออกจากโหมดโกรธเร็วขึ้น
        else:
            if R_att >= 0.75: state_mode = "HAPPY"
            elif state_mode == "HAPPY" and R_att >= 0.55: state_mode = "HAPPY"
            elif R_att >= 0.35: state_mode = "CURIOUS"
            elif state_mode == "CURIOUS" and R_att >= 0.20: state_mode = "CURIOUS"
            else:
                if state_mode == "SLEEPY":
                    if R_rst <= 0.15: state_mode = "EXPLORING"
                else:
                    if R_rst >= 0.90: state_mode = "SLEEPY"
                    elif R_rst >= 0.75: state_mode = "NORMAL"
                    elif R_rst >= 0.55: state_mode = "SAD"
                    else: state_mode = "EXPLORING"

        led_rgb = LED_COLORS.get(state_mode, (220, 220, 220))
        if sulking:
            led_rgb = LED_COLORS["SULK"]
        if quiet_phase is not None:
            led_rgb = LED_COLORS["QUIET"]

        # 5. สั่งการท่าทาง (ท่าชู่ว์มาก่อน -> งอน -> อารมณ์อื่น)
        #    ท่าชู่ว์ใช้ scale=False: ไม่ให้ mood มายืดหรือเร่งจังหวะพยักหน้า
        if quiet_phase == "BACK":
            # ถอยหลัง ยืดหลังตรง (ทุกค่าอยู่ในลิมิต ANGLE_LIM)
            set_auto_movej([quiet_yaw, QUIET_BACK_Q[1], QUIET_BACK_Q[2], QUIET_BACK_Q[3]], duration=0.30, spd=30, scale=False)
        elif quiet_phase == "NOD":
            # พยักหน้า: ก้ม -> เงย ตามจำนวน NOD_COUNT ครั้ง
            p = (now - quiet_t0 - QUIET_BACK_T) / QUIET_NOD_T
            nod = 0.5 * (1.0 - math.cos(2.0 * math.pi * NOD_COUNT * p))
            set_auto_movej([quiet_yaw, QUIET_BACK_Q[1] - 4.0 * nod, QUIET_BACK_Q[2], QUIET_BACK_Q[3] + NOD_AMP * nod],
                           duration=0.10, spd=40, scale=False)
        elif quiet_phase == "HUSH":
            # เงียบ: นิ่ง ไม่ส่าย ก้มหัวลงนิด ๆ ไฟหรี่
            set_auto_movej([quiet_yaw, QUIET_BACK_Q[1] - 3.0, QUIET_BACK_Q[2], QUIET_HUSH_PITCH], duration=0.40, spd=10, scale=False)
        elif sulking:
            # งอน: หันหน้าหนีคน ก้มหน้านิด ๆ ไม่สนใจ
            # หลอดใกล้หาย (โดนง้อ) -> แอบหันกลับมามองนิด ๆ
            person_yaw = (0.5 - cx_filt + CAM_PAN_OFFSET) * 110.0
            away_yaw = person_yaw + sulk_side * 75.0
            peek = clamp((0.6 - sulk) / 0.3, 0.0, 1.0) * 0.4
            sulk_yaw = clamp(away_yaw + (person_yaw - away_yaw) * peek, -75.0, 75.0)
            set_auto_movej([sulk_yaw + 2.0 * o1, 88.0, -85.0, 22.0 + 3.0 * o2], duration=0.40, spd=15)
        elif state_mode == "STARTLED":
            yaw_deg = clamp((0.5 - cx_filt + CAM_PAN_OFFSET) * 95.0, -70.0, 70.0)
            q_sol = solve_ik(85.0 * math.cos(math.radians(yaw_deg)), 85.0 * math.sin(math.radians(yaw_deg)), 245.0, -22.0, current_q[0])
            set_auto_movej(q_sol if q_sol else [yaw_deg, 115.0, -112.0, -22.0], duration=0.14, spd=60)
        elif state_mode == "ANGRY":
            yaw_deg = (0.5 - cx_filt + CAM_PAN_OFFSET) * 90.0
            angry_yaw = clamp(yaw_deg + 28.0 * o1, -75.0, 75.0)
            angry_shoulder = 62.0 + 12.0 * o2
            angry_elbow = -55.0 - 12.0 * o2
            angry_pitch = clamp(-5.0 + 22.0 * o1, -40.0, 38.0)
            set_auto_movej([angry_yaw, angry_shoulder, angry_elbow, angry_pitch], duration=0.12, spd=55)
        elif state_mode == "HAPPY":
            # wv = 0 ท่าดีใจปกติ, wv = 1 เห็นโบกมือ -> ส่ายไปมากว้างขึ้น เร็วขึ้น
            wv = wave_level
            yaw_deg = (0.5 - cx_filt + CAM_PAN_OFFSET) * 110.0
            happy_yaw = clamp(yaw_deg + (8.0 + 27.0 * wv) * o1, -75.0, 75.0)
            happy_shoulder = 75.0 + (6.0 + 10.0 * wv) * o1
            pitch_cmd = (cy_filt - 0.5 + CAM_TILT_OFFSET) * 35.0 + ((12.0 + 10.0 * wv) * o2) - 20.0
            set_auto_movej([happy_yaw, happy_shoulder, -40.0, clamp(pitch_cmd, -42.0, 38.0)],
                           duration=0.20 - 0.06 * wv, spd=int(30 + 20 * wv))
        elif state_mode == "CURIOUS":
            yaw_deg = (0.5 - cx_filt + CAM_PAN_OFFSET) * 110.0
            tx, ty = 135.0 * math.cos(math.radians(yaw_deg)), 135.0 * math.sin(math.radians(yaw_deg))
            pitch_cmd = (cy_filt - 0.5 + CAM_TILT_OFFSET) * 40.0
            z_target = 230.0 - (cy_filt - 0.5) * 50.0
            q_sol = solve_ik(tx, ty, z_target, pitch_cmd, current_q[0])
            set_auto_movej(q_sol if q_sol else [yaw_deg, 80.0, -75.0, pitch_cmd], duration=0.25, spd=25)
        elif state_mode == "SAD":
            set_auto_movej([12.0 * o1, 95.0 + 3.0 * o2, -95.0, 32.0 + 4.0 * o2], duration=0.30, spd=20)
        elif state_mode == "NORMAL":
            set_auto_movej([5.0 * o1, 80.0, -75.0, -10.0 + 2.0 * o2], duration=0.30, spd=20)
        elif state_mode == "SLEEPY":
            set_auto_movej([0.0, 102.0 + 1.5 * o1, -102.0, 36.0 + 2.0 * o2], duration=0.35, spd=15)
        else:
            set_auto_movej([45.0 * o1, 70.0, -60.0, -30.0 + 10.0 * o2], duration=0.25, spd=25)

        disp_mode = f"QUIET:{quiet_phase}" if quiet_phase else ("SULK" if sulking else state_mode)
        graph_panel = draw_hormone_graph(hist_att, hist_ang, hist_str, hist_rst, disp_mode, gesture_name, phi_cmd,
                                         mood=mood, sulk=sulk, speed=speed_mult)
        view_frame = cv2.resize(frame, (graph_panel.shape[1] * 480 // 460, graph_panel.shape[0]))   # ขยายภาพกล้องให้สูงเท่ากราฟ
        combined_view = np.hstack((view_frame, graph_panel))
        cv2.imshow(WIN_NAME, combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key in (ord('+'), ord('=')): set_graph_scale(GRAPH_SCALE + 0.1)   # กด + = ขยาย
        if key in (ord('-'), ord('_')): set_graph_scale(GRAPH_SCALE - 0.1)   # กด - = ย่อ
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1: break   # กดปุ่ม X ปิดหน้าต่าง

        sleep_rem = dt - (time.time() - loop_start)
        if sleep_rem > 0: time.sleep(sleep_rem)

def main():
    global stop_threads
    cap = open_camera()   # เปิดกล้องก่อน ถ้าไม่มีกล้องจะได้ไม่ต้องปลุกแขนขึ้นมาเปล่า ๆ
    if cap is None:
        if ser: ser.close()
        return

    startup_sequence()   # ท่าตอนเปิด
    t_motion = threading.Thread(target=motion_control_thread, daemon=True)
    t_motion.start()

    try:
        main_loop(cap)
    except KeyboardInterrupt:
        print("\n⌨️ กด Ctrl+C")
    finally:
        # ไม่ว่าปิดด้วยวิธีไหน (q / ปุ่ม X / Ctrl+C / error) จะเข้าท่าปิดเสมอ
        stop_threads = True
        t_motion.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        shutdown_sequence()
        try:
            hands_detector.close()
            face_detector.close()
        except Exception:
            pass
        if ser: ser.close()
        print("🛑 ปิดระบบเรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()
