import pyautogui
from pynput import mouse
import time

# Function to wait for a mouse click and return its coordinates
def wait_for_click():
    click_pos = None
    def on_click(x, y, button, pressed):
        nonlocal click_pos
        if pressed and button == mouse.Button.left:
            click_pos = (x, y)
            return False  # Stop listener
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    return click_pos

# Define the game window region dynamically
print("Please click on the top-left corner of the game window.")
time.sleep(2)  # Give you time to move the mouse
top_left = wait_for_click()
print(f"Top-left corner clicked at: {top_left}")

print("Now, click on the bottom-right corner of the game window.")
time.sleep(2)  # Give you time again
bottom_right = wait_for_click()
print(f"Bottom-right corner clicked at: {bottom_right}")

# Validate the clicks
if bottom_right[0] <= top_left[0] or bottom_right[1] <= top_left[1]:
    print("Invalid clicks. Bottom-right must be below and to the right of top-left.")
    exit()

# Calculate game window region
GAME_WINDOW_LEFT = top_left[0]
GAME_WINDOW_TOP = top_left[1]
GAME_WINDOW_WIDTH = bottom_right[0] - top_left[0]
GAME_WINDOW_HEIGHT = bottom_right[1] - top_left[1]

GAME_WINDOW_REGION = {
    'top': GAME_WINDOW_TOP,
    'left': GAME_WINDOW_LEFT,
    'width': GAME_WINDOW_WIDTH,
    'height': GAME_WINDOW_HEIGHT
}

# Display the calculated dimensions
print("\nGame Window Dimensions:")
print(f"Top: {GAME_WINDOW_REGION['top']}")
print(f"Left: {GAME_WINDOW_REGION['left']}")
print(f"Width: {GAME_WINDOW_REGION['width']}")
print(f"Height: {GAME_WINDOW_REGION['height']}")

# Optional: Test the region by capturing a screenshot
input("Press Enter to take a test screenshot of the defined region (or Ctrl+C to exit)...")
screenshot = pyautogui.screenshot(region=(GAME_WINDOW_LEFT, GAME_WINDOW_TOP, GAME_WINDOW_WIDTH, GAME_WINDOW_HEIGHT))
screenshot.save("test_game_window.png")