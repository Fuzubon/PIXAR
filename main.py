import time
import sys
from machine import UART
from fpioa_manager import fm
from modules import ws2812

fm.register(24, fm.fpioa.UART2_TX, force=True)
uart_servo = UART(UART.UART2, 1000000, 8, None, 1, timeout=10)
ring = ws2812(14, 12)

def set_all_leds(r, g, b):
    for i in range(12): ring.set_led(i, (r, g, b))
    ring.display()

def enable_torque(servo_id, enable):
    val = 1 if enable else 0
    checksum = ~(servo_id + 4 + 0x03 + 0x28 + val) & 0xFF
    uart_servo.write(bytearray([0xFF, 0xFF, servo_id, 4, 0x03, 0x28, val, checksum]))

def move_servo(servo_id, pos, speed_percent):
    pos = max(0, min(1023, int(pos)))
    pos_l = pos & 0xFF
    pos_h = (pos >> 8) & 0xFF
    speed_raw = int((speed_percent / 100.0) * 1000)
    v_h = (speed_raw >> 8) & 0xFF
    v_l = speed_raw & 0xFF
    checksum = ~(servo_id + 9 + 0x03 + 0x2A + pos_h + pos_l + 0 + 0 + v_h + v_l) & 0xFF
    packet = bytearray([0xFF, 0xFF, servo_id, 9, 0x03, 0x2A, pos_h, pos_l, 0, 0, v_h, v_l, checksum])
    uart_servo.write(packet)

for i in [3, 4, 5, 6]:
    enable_torque(i, 1)

# ไฟเช็กสถานะการบูต: เขียว -> น้ำเงิน = บูตสมบูรณ์พร้อมรับคำสั่ง
set_all_leds(0, 255, 0)
time.sleep(1)
set_all_leds(0, 0, 255)
time.sleep(1)
set_all_leds(0, 0, 0)

last_tq = 1

# ลูปรับข้อมูลแบบดิบๆ (เสถียรสุด)
while True:
    try:
        line = sys.stdin.readline()
        if line:
            parts = line.strip().split(',')
            if len(parts) == 9:
                p4, p5, p6, p3, spd, r, g, b, tq = [int(x) for x in parts]

                if tq != last_tq:
                    for i in [3, 4, 5, 6]: enable_torque(i, tq)
                    last_tq = tq

                if tq == 1:
                    move_servo(4, p4, spd)
                    move_servo(5, p5, spd)
                    move_servo(6, p6, spd)
                    move_servo(3, p3, spd)

                set_all_leds(r, g, b)
    except Exception:
        pass
