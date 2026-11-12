import time
from machine import UART
from fpioa_manager import fm

# 1. ตั้งค่าพอร์ต
fm.register(24, fm.fpioa.UART2_TX, force=True)
uart_servo = UART(UART.UART2, 1000000, 8, None, 1, timeout=10)

# ==========================================
# 🛠️ ตั้งค่าสำหรับทดสอบหาลิมิตทีละตัว
# ==========================================
TEST_ID = 6         # เปลี่ยนเลขตรงนี้เพื่อทดสอบ (3, 5, หรือ 6)
START_POS = 512     # จุดตั้งต้น (ถ้าทดสอบ ID 5 ให้ใช้ 750 / ถ้า ID 3, 6 ให้ใช้ 512)
DIRECTION = -1       # ใส่ 1 เพื่อกวาดหาค่ามาก (Max) / ใส่ -1 เพื่อกวาดหาค่าน้อย (Min)
# ==========================================

def enable_torque(servo_id, enable):
    val = 1 if enable else 0
    checksum = ~(servo_id + 4 + 0x03 + 0x28 + val) & 0xFF
    uart_servo.write(bytearray([0xFF, 0xFF, servo_id, 4, 0x03, 0x28, val, checksum]))

def set_torque_limit(servo_id, torque):
    torque = max(0, min(1023, torque))
    t_l = torque & 0xFF
    t_h = (torque >> 8) & 0xFF
    checksum = ~(servo_id + 5 + 0x03 + 0x30 + t_h + t_l) & 0xFF
    uart_servo.write(bytearray([0xFF, 0xFF, servo_id, 5, 0x03, 0x30, t_h, t_l, checksum]))

def move_servo(servo_id, pos):
    pos = max(0, min(1023, pos))
    pos_l = pos & 0xFF
    pos_h = (pos >> 8) & 0xFF
    t_h = 0x01; t_l = 0xF4 # 500ms
    checksum = ~(servo_id + 9 + 0x03 + 0x2A + pos_h + pos_l + t_h + t_l + 0 + 0) & 0xFF
    uart_servo.write(bytearray([0xFF, 0xFF, servo_id, 9, 0x03, 0x2A, pos_h, pos_l, t_h, t_l, 0, 0, checksum]))

print("=== Calibration Mode ===")

# 2. ปลุกมอเตอร์
for i in [3, 4, 5, 6]:
    enable_torque(i, 1)
    time.sleep_ms(50)
    set_torque_limit(i, 500) 
    time.sleep_ms(50)
    
# จัดตัวอื่นให้อยู่นิ่งๆ
if TEST_ID != 4: move_servo(4, 512)
if TEST_ID != 5: move_servo(5, 750)
if TEST_ID != 6: move_servo(6, 512)
if TEST_ID != 3: move_servo(3, 512)
time.sleep(1)

# ขยับตัวที่ต้องการทดสอบไปจุดเริ่มต้น
move_servo(TEST_ID, START_POS)
print("จัดท่ามอเตอร์ {} ไปที่ {}".format(TEST_ID, START_POS))
time.sleep(2)

dir_text = "บวก (+)" if DIRECTION == 1 else "ลบ (-)"
print("เตรียมตัว! กำลังกวาดไปทาง {}".format(dir_text))
print("!!! เอามือจับเมาส์รอไว้ ถ้าใกล้ชนให้กดปุ่ม Stop (สีแดง) ทันที !!!")
time.sleep(2)

current_pos = START_POS
while (0 <= current_pos <= 1023):
    move_servo(TEST_ID, current_pos)
    print("ID {} Current Angle = {}".format(TEST_ID, current_pos))
    
    current_pos += (10 * DIRECTION)
    time.sleep_ms(500)