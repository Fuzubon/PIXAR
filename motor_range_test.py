import time
from machine import UART
from fpioa_manager import fm

fm.register(24, fm.fpioa.UART2_TX, force=True)
uart_servo = UART(UART.UART2, 1000000, 8, None, 1, timeout=10)

def enable_torque(servo_id, enable):
    val = 1 if enable else 0
    checksum = ~(servo_id + 4 + 0x03 + 0x28 + val) & 0xFF
    uart_servo.write(bytearray([0xFF, 0xFF, servo_id, 4, 0x03, 0x28, val, checksum]))

def move_servo_speed_percent(servo_id, pos, speed_percent=30):
    pos = max(0, min(1023, pos))
    pos_l = pos & 0xFF
    pos_h = (pos >> 8) & 0xFF
    
    speed_raw = int((speed_percent / 100.0) * 1000)
    v_h = (speed_raw >> 8) & 0xFF
    v_l = speed_raw & 0xFF
    
    t_h = 0
    t_l = 0
    
    checksum = ~(servo_id + 9 + 0x03 + 0x2A + pos_h + pos_l + t_h + t_l + v_h + v_l) & 0xFF
    packet = bytearray([0xFF, 0xFF, servo_id, 9, 0x03, 0x2A, pos_h, pos_l, t_h, t_l, v_h, v_l, checksum])
    uart_servo.write(packet)

print("=== Test All Motors within Hardware Limits ===")

# 1. เปิด Torque ทุกตัว
for i in [3, 4, 5, 6]:
    enable_torque(i, 1)
    time.sleep_ms(50)

# กำหนดความเร็ว 30% ให้เคลื่อนไหวสมูทและมีแรงดึงพอ
TEST_SPEED = 30

# 2. ขยับไปที่ตำแหน่งใกล้ 0 (แต่เผื่อระยะไว้ที่ 100)
print("Action 1: ขยับทุกตัวไปที่ตำแหน่ง 100")
move_servo_speed_percent(4, 600, TEST_SPEED) 
move_servo_speed_percent(5, 100, TEST_SPEED) 
move_servo_speed_percent(6, 100, TEST_SPEED) 
move_servo_speed_percent(3, 300, TEST_SPEED) 
time.sleep(3) # รอให้เดินทางถึง 3 วินาที

# 3. ขยับไปที่ตำแหน่งเป้าหมายใกล้ขีดจำกัด Max ของแต่ละตัวที่คุณตั้งไว้
print("Action 2: ขยับกางออกใกล้ขีดจำกัดสูงสุดของแต่ละตัว")
move_servo_speed_percent(4, 375, TEST_SPEED) # (Limit 4 = 810)
move_servo_speed_percent(5, 600, TEST_SPEED) # (Limit 5 = 650)
move_servo_speed_percent(6, 550, TEST_SPEED) # (Limit 6 = 600)
move_servo_speed_percent(3, 025, TEST_SPEED) # (Limit 3 = 450)
time.sleep(3) # รอให้เดินทางถึง 3 วินาที

print("จบการทดสอบ ล็อกสถานะค้างไว้ (กด Stop สีแดงเพื่อหยุด)")
# 4. ลูปค้างสถานะ ป้องกันโปรแกรมตัดจบก่อนมอเตอร์หมุนเสร็จ
while True:
    time.sleep(1)