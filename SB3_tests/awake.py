import pyautogui
import time
import random

print("Starting mouse activity script...")
print("Press Ctrl+C to stop")

# Disable pyautogui's fail-safe (optional - be careful)
# pyautogui.FAILSAFE = False

try:
    while True:
        # Get current mouse position
        current_x, current_y = pyautogui.position()
        
        # Small random movement (1-3 pixels)
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        
        # Move mouse slightly
        pyautogui.moveRel(dx, dy)
        
        # Move back to original position
        time.sleep(0.1)
        pyautogui.moveTo(current_x, current_y)
        
        # Occasionally do a gentle click (every 3rd iteration)
        if random.randint(1, 3) == 1:
            # Click at current position (very gentle)
            pyautogui.click(current_x, current_y)
            print(f"Gentle click at ({current_x}, {current_y})")
        else:
            print(f"Mouse jiggle at ({current_x}, {current_y})")
        
        # Wait 45-75 seconds (random interval)
        wait_time = random.randint(45, 75)
        print(f"Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        
except KeyboardInterrupt:
    print("\nMouse activity script stopped.")
except Exception as e:
    print(f"Error: {e}")