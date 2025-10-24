import pyautogui as pg
import time

pg.FAILSAFE = True  # move mouse to corner to abort

# config
delay = 30         # seconds between clicks
x, y = 800, 450      # click position (set yours)
infinite = True


time.sleep(3)

try:
    while infinite:
        pg.click(x=x, y=y)
        time.sleep(delay)
except KeyboardInterrupt:
    print("\nStopped by user.")
