import cv2
import numpy as np
import mss
import pyautogui
import time

# Screen resolution (single monitor)
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# Define the full screen region
SCREEN_REGION = {'top': 0, 'left': 0, 'width': SCREEN_WIDTH, 'height': SCREEN_HEIGHT}

# Load templates
KNIGHT_TEMPLATE = cv2.imread('images/knight.png')
KNIGHT_QUESTION1 = cv2.imread('images/knight_question1.png')
SELECT1 = cv2.imread('images/select1.png')
KNIGHT_QUESTION2 = cv2.imread('images/knight_question2.png')
SELECT2 = cv2.imread('images/select2.png')
EXIT_TEMPLATE = cv2.imread('images/exit.png')  # New exit template
CHECK_FOR_ROOM2 = cv2.imread('images/check_for_room2.png')
PORTAL_TEMPLATE = cv2.imread('images/portal.png')
CHARACTER_LEFT = cv2.imread('images/character_left.png')
CHARACTER_RIGHT = cv2.imread('images/character_right.png')
CHECK_FOR_ROOM3 = cv2.imread('images/check_for_room3.png')
CHECK_FOR_ROOM4 = cv2.imread('images/check_for_room4.png')
CHECK_FOR_ROOM3_END = cv2.imread('images/room3_end.png')

ROOM2_WOLF_TEMPLATES = [
    cv2.imread('images/room2_wolf1.png'),
    cv2.imread('images/room2_wolf2.png'),
    cv2.imread('images/room2_wolf3.png'),
    cv2.imread('images/room2_wolf4.png'),
    cv2.imread('images/room2_wolf5.png'),
]

# Check if templates loaded successfully
for name, template in [
    ("knight.png", KNIGHT_TEMPLATE),
    ("knight_question1.png", KNIGHT_QUESTION1),
    ("select1.png", SELECT1),
    ("knight_question2.png", KNIGHT_QUESTION2),
    ("select2.png", SELECT2),
    ("exit.png", EXIT_TEMPLATE),
    ("check_for_room2.png", CHECK_FOR_ROOM2),
    ("portal.png", PORTAL_TEMPLATE),
    ("character_left.png", CHARACTER_LEFT),
    ("character_right.png", CHARACTER_RIGHT),
    ("check_for_room3.png", CHECK_FOR_ROOM3),
    ("check_for_room3_end.png", CHECK_FOR_ROOM3_END)

]:
    if template is None:
        print(f"Error: Could not load '{name}'.")
        exit()

for i, template in enumerate(ROOM2_WOLF_TEMPLATES, 1):
    if template is None:
        print(f"Error: Could not load 'room2_wolf{i}.png'.")
        exit()

def capture_screen(region):
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def find_template(screen, template, threshold=0.5):
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f"Detection - Confidence: {max_val:.3f}")
    if max_val > threshold:
        center_x = max_loc[0] + (template.shape[1] // 2)
        center_y = max_loc[1] + (template.shape[0] // 2)
        top_left_x = max_loc[0]
        top_left_y = max_loc[1]
        width = template.shape[1]
        height = template.shape[0]
        return center_x, center_y, top_left_x, top_left_y, width, height, max_val
    return None, None, None, None, None, None, max_val

def double_jump_left():
    for _ in range(1):
        pyautogui.press('c')
        pyautogui.press('c')
        pyautogui.keyDown('left')
        pyautogui.sleep(0.1)
        pyautogui.keyUp('left')
        time.sleep(0.5)

def knight_interaction():
    print("Starting Initial Knight Interaction...")
    step = "knight"
    dialog_region = None
    knight_found = False
    knight_x, knight_y = None, None  # Store knight coordinates

    while True:
        screen = capture_screen(SCREEN_REGION)

        if step == "knight":
            print("Searching for knight...")
            x, y, _, _, _, _, confidence = find_template(screen, KNIGHT_TEMPLATE)
            if x is not None and y is not None:
                print(f"Knight found at: ({x}, {y})")
                knight_x, knight_y = x, y  # Store knight coordinates
                pyautogui.moveTo(x, y)
                pyautogui.click()
                time.sleep(0.5)
                print("Knight confirmed! Moving to next step.")
                knight_found = True
                step = "knight_question1"
            else:
                print("Knight not found. Performing 2 double jumps left...")
                double_jump_left()

        elif step == "knight_question1":
            print("Searching for knight_question1...")
            x, y, top_left_x, top_left_y, width, height, confidence = find_template(screen, KNIGHT_QUESTION1)
            if x is not None and y is not None:
                print(f"Knight question 1 found at: ({x}, {y})")
                print("Knight question 1 confirmed! Moving to select1.")
                dialog_region = {
                    'top': top_left_y,
                    'left': top_left_x,
                    'width': width,
                    'height': height
                }
                step = "select1"
            else:
                print("Knight question 1 not found. Retrying...")

        elif step == "select1":
            print("Searching for select1 within knight_question1 dialog...")
            if dialog_region is None:
                print("No dialog region defined. Moving to next step.")
                step = "knight_question2"
                continue
            start_time = time.time()
            while time.time() - start_time < 1:  # 1-second timeout
                screen = capture_screen(dialog_region)
                x, y, _, _, _, _, confidence = find_template(screen, SELECT1, threshold=0.4)
                if x is not None and y is not None:
                    screen_x = x + dialog_region['left']
                    screen_y = y + dialog_region['top']
                    print(f"Select1 found at: ({screen_x}, {screen_y})")
                    pyautogui.moveTo(screen_x, screen_y)
                    pyautogui.click()
                    time.sleep(0.5)
                    print("Select1 confirmed! Moving to knight_question2.")
                    step = "knight_question2"
                    break
                time.sleep(0.1)
            else:
                print("Select1 not found within 2 seconds. Searching for exit button...")
                screen = capture_screen(dialog_region)
                x, y, _, _, _, _, confidence = find_template(screen, EXIT_TEMPLATE, threshold=0.4)
                if x is not None and y is not None:
                    screen_x = x + dialog_region['left']
                    screen_y = y + dialog_region['top']
                    print(f"Exit button found at: ({screen_x}, {screen_y})")
                    pyautogui.moveTo(screen_x, screen_y)
                    pyautogui.click()
                    time.sleep(0.5)
                    print("Exit button clicked. Reclicking knight to restart...")
                    if knight_x is not None and knight_y is not None:
                        pyautogui.moveTo(knight_x, knight_y)
                        pyautogui.click()
                        time.sleep(0.5)
                        step = "knight_question1"  # Restart from question 1
                    else:
                        print("Knight coordinates not available. Moving to next step.")
                        step = "knight_question2"
                else:
                    print("Exit button not found. Moving to next step.")
                    step = "knight_question2"

        elif step == "knight_question2":
            print("Searching for knight_question2...")
            x, y, top_left_x, top_left_y, width, height, confidence = find_template(screen, KNIGHT_QUESTION2)
            if x is not None and y is not None:
                print(f"Knight question 2 found at: ({x}, {y})")
                print("Knight question 2 confirmed! Moving to select2.")
                dialog_region = {
                    'top': top_left_y,
                    'left': top_left_x,
                    'width': width,
                    'height': height
                }
                step = "select2"
            else:
                print("Knight question 2 not found. Retrying...")

        elif step == "select2":
            print("Searching for select2 within knight_question2 dialog...")
            if dialog_region is None:
                print("No dialog region defined. Moving to missions.")
                break
            start_time = time.time()
            while time.time() - start_time < 1:  # 1-second timeout
                screen = capture_screen(dialog_region)
                x, y, _, _, _, _, confidence = find_template(screen, SELECT2, threshold=0.4)
                if x is not None and y is not None:
                    screen_x = x + dialog_region['left']
                    screen_y = y + dialog_region['top']
                    print(f"Select2 found at: ({screen_x}, {screen_y})")
                    pyautogui.moveTo(screen_x, screen_y)
                    pyautogui.click()
                    time.sleep(0.5)
                    print("Select2 confirmed! Moving to missions.")
                    return  # Exit the function successfully
                time.sleep(0.1)
            else:
                print("Select2 not found within 2 seconds. Searching for exit button...")
                screen = capture_screen(dialog_region)
                x, y, _, _, _, _, confidence = find_template(screen, EXIT_TEMPLATE, threshold=0.4)
                if x is not None and y is not None:
                    screen_x = x + dialog_region['left']
                    screen_y = y + dialog_region['top']
                    print(f"Exit button found at: ({screen_x}, {screen_y})")
                    pyautogui.moveTo(screen_x, screen_y)
                    pyautogui.click()
                    time.sleep(0.5)
                    print("Exit button clicked. Reclicking knight to restart...")
                    if knight_x is not None and knight_y is not None:
                        pyautogui.moveTo(knight_x, knight_y)
                        pyautogui.click()
                        time.sleep(0.5)
                        step = "knight_question1"  # Restart from question 1
                    else:
                        print("Knight coordinates not available. Moving to missions.")
                        break
                else:
                    print("Exit button not found. Moving to missions.")
                    break

        time.sleep(0.1)


def align_with_object(target_template, threshold=0.5, tolerance=10):
    print("Aligning character with target object...")
    while True:
        screen = capture_screen(SCREEN_REGION)

        # Detect the character (either facing left or right)
        character_x = None
        x, y, _, _, _, _, confidence = find_template(screen, CHARACTER_LEFT, threshold=0.5)
        if x is not None and y is not None:
            print(f"Character (left) detected at: ({x}, {y}) with confidence {confidence:.3f}")
            character_x = x
        else:
            x, y, _, _, _, _, confidence = find_template(screen, CHARACTER_RIGHT, threshold=0.5)
            if x is not None and y is not None:
                print(f"Character (right) detected at: ({x}, {y}) with confidence {confidence:.3f}")
                character_x = x

        # Detect the target object (portal)
        target_x = None
        x, y, _, _, _, _, confidence = find_template(screen, target_template, threshold=threshold)
        if x is not None and y is not None:
            print(f"Target object detected at: ({x}, {y}) with confidence {confidence:.3f}")
            target_x = x

        # If either character or target isn't found, retry
        if character_x is None or target_x is None:
            print("Character or target not detected. Retrying...")
            time.sleep(0.5)
            continue

        # Calculate horizontal distance between character and target
        distance = target_x - character_x
        print(f"Horizontal distance between character and target: {distance}")

        # Check if aligned (within tolerance)
        if abs(distance) <= tolerance:
            print("Character aligned with target object!")
            break

        # Move left or right based on distance
        if distance < 0:
            print("Target is to the left. Moving left twice...")
            for _ in range(2):
                pyautogui.keyDown('left')
                time.sleep(0.2)
                pyautogui.keyUp('left')
                time.sleep(0.2)
        else:
            print("Target is to the right. Moving right twice...")
            for _ in range(2):
                pyautogui.keyDown('right')
                time.sleep(0.2)
                pyautogui.keyUp('right')
                time.sleep(0.2)

        time.sleep(0.5)  # Delay before rechecking

# Mission Functions
def mission_1():
    print("Mission 1: Starting...")
    print("Waiting 10 seconds before starting Mission 1 tasks...")
    time.sleep(10)  # Wait 10 seconds before starting
    print("Mission 1: Checking for Room 2 map...")

    # Continuously check for the Room 2 map
    while True:
        screen = capture_screen(SCREEN_REGION)
        x, y, _, _, _, _, confidence = find_template(screen, CHECK_FOR_ROOM2, threshold=0.5)
        if x is not None and y is not None:
            print(f"Room 2 map detected at: ({x}, {y}) with confidence {confidence:.3f}")
            break
        print("Room 2 map not detected. Retrying...")
        time.sleep(0.5)  # Check every 0.5 seconds

    print("Mission 1: In Room 2. Executing initial commands...")

    # Perform down + c three times (simultaneous down and double c)
    for _ in range(3):
        pyautogui.keyDown('down')
        pyautogui.press('c')  # First c press
        pyautogui.press('c')  # Second c press for double jump
        pyautogui.keyUp('down')
        time.sleep(0.2)  # Small delay between actions

    # Click right arrow and press z two times
    pyautogui.keyDown('right')
    time.sleep(0.2)
    pyautogui.keyUp('right')
    time.sleep(0.2)
    pyautogui.press('z')
    time.sleep(0.2)
    pyautogui.press('z')
    time.sleep(0.2)

    print("Mission 1: Initial commands executed. Starting jump right and z sequence...")

    pyautogui.keyDown('alt')
    time.sleep(2)  # Hold Alt for 2 seconds
    pyautogui.keyUp('alt')
    time.sleep(0.5)  # Delay to ensure the game registers the input

    # First jump right and z sequence (double jump with right)
    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)  # Delay between jump sequences

    pyautogui.keyDown('right')
    pyautogui.press('c')
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)  # Delay between jump sequences

    pyautogui.keyDown('right')
    pyautogui.press('c')
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)

    pyautogui.keyDown('right')
    pyautogui.press('c')
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)  # Delay between jump sequences

    print("Mission 1: Holding z for 10 seconds...")

    # Hold z for 10 seconds (press z repeatedly to simulate holding)
    attack_start = time.time()
    while time.time() - attack_start < 12:
        pyautogui.press('z')
        time.sleep(0.001)  # Small delay between z presses

    print("Mission 1: Repeating jump right and z sequence...")

    # Repeat the jump right and z sequence (double jump with right)
    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)  # Delay between jump sequences

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)  # Delay between jump sequences

    pyautogui.keyDown('left')
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)  # Delay between jump sequences


    pyautogui.keyDown('left')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('left')
    time.sleep(0.5)  # Delay between jump sequences

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)  # Delay between jump sequences

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)  # Delay between jump sequences

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)

    print("Mission 1: Performing final movement sequence to reach portal...")


    pyautogui.keyDown('left')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('left')
    time.sleep(0.5)

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)

    pyautogui.keyDown('right')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('right')
    time.sleep(0.5)


    pyautogui.keyDown('down')
    pyautogui.keyDown('down')
    pyautogui.press('b')  # First c press
    pyautogui.press('b')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('down')
    time.sleep(0.5)

    pyautogui.keyDown('up')
    pyautogui.keyDown('up')
    pyautogui.press('b')  # First c press
    pyautogui.press('b')  # Second c press for double jump
    for _ in range(10):
        pyautogui.press('z')
        time.sleep(0.01)
    pyautogui.keyUp('up')
    time.sleep(0.5)

    # Turn left
    pyautogui.keyDown('left')
    time.sleep(0.2)
    pyautogui.keyUp('left')
    time.sleep(0.2)

    # Press tab
    pyautogui.keyDown('b')
    time.sleep(0.5)  # Hold b for 0.5 seconds
    pyautogui.keyUp('b')
    time.sleep(0.5)  # Delay to ensure the game registers the input

    # Move right 6 times
    for _ in range(4):
        pyautogui.keyDown('left')
        time.sleep(0.2)
        pyautogui.keyUp('left')
        time.sleep(0.2)

    # Press up to interact with the portal
    pyautogui.press('up')
    pyautogui.press('up')
    pyautogui.press('up')
    time.sleep(0.5)  # Delay to ensure the game registers the input

    print("Mission 1: Completed!")


def mission_2():
    print("Mission 2: Starting...")
    print("Mission 2: Waiting for Room 3 map...")

    # Continuously check for the Room 3 map
    while True:
        screen = capture_screen(SCREEN_REGION)
        x, y, _, _, _, _, confidence = find_template(screen, CHECK_FOR_ROOM3, threshold=0.5)
        if x is not None and y is not None:
            print(f"Room 3 map detected at: ({x}, {y}) with confidence {confidence:.3f}")
            break
        print("Room 3 map not detected. Retrying...")
        time.sleep(0.5)  # Check every 0.5 seconds

    print("Mission 2: In Room 3. Executing double jump left...")

    # Double jump left
    pyautogui.keyDown('left')
    pyautogui.press('c')  # First c press
    pyautogui.press('c')  # Second c press for double jump
    pyautogui.keyUp('left')
    time.sleep(0.5)  # Delay after action

    print("Mission 2: Holding down for 6 seconds while pressing c every 0.5 seconds...")

    # Hold down for 6 seconds, pressing c every 0.5 seconds
    pyautogui.keyDown('down')
    start_time = time.time()
    while time.time() - start_time < 6:
        pyautogui.press('c')
        time.sleep(0.5)  # Press c every 0.5 seconds
    pyautogui.keyUp('down')
    time.sleep(0.5)  # Delay after action

    print("Mission 2: Turning right and performing 4 single jumps right...")

    # Turn right and perform 4 single jumps right
    pyautogui.keyDown('right')
    time.sleep(0.2)  # Brief turn right
    pyautogui.keyUp('right')
    time.sleep(0.2)
    for _ in range(4):
        pyautogui.keyDown('right')
        pyautogui.press('c')  # Single jump
        pyautogui.keyUp('right')
        time.sleep(0.5)  # Delay between jumps

    print("Mission 2: Pressing 2 and performing 2 double jumps right with 3 comma presses after each...")

    # Press 2, then perform 2 double jumps right with 3 comma presses after each
    pyautogui.press('2')
    time.sleep(0.5)  # Delay to ensure game registers 2
    for _ in range(2):
        pyautogui.keyDown('right')
        pyautogui.press('c')  # First c press
        pyautogui.press('c')  # Second c press for double jump
        pyautogui.keyUp('right')
        time.sleep(0.2)  # Brief delay before comma presses
        for _ in range(3):
            pyautogui.press('comma')  # Press comma key (replace with correct key if needed)
            time.sleep(0.2)  # Delay between comma presses
        time.sleep(0.5)  # Delay between jumps

    print("Mission 2: Turning left and holding comma for 5 seconds...")

    # Turn left and hold comma for 5 seconds
    pyautogui.keyDown('left')
    time.sleep(0.2)  # Brief turn left
    pyautogui.keyUp('left')
    time.sleep(0.2)
    pyautogui.keyDown('comma')
    time.sleep(5)  # Hold comma for 5 seconds
    pyautogui.keyUp('comma')
    time.sleep(0.5)  # Delay after action

    print("Mission 2: Pressing 2 again and holding z for 5 seconds...")

    # Press 2 again, then hold z for 5 seconds
    pyautogui.press('2')
    time.sleep(0.5)  # Delay to ensure game registers 2
    pyautogui.keyDown('z')
    time.sleep(5)  # Hold z for 5 seconds
    pyautogui.keyUp('z')
    time.sleep(0.5)  # Delay after action

    print("Mission 2: Looping double jump left and right with up presses until room2_end detected...")

    # Loop double jump left and right with up presses until room2_end.png is detected
    while True:
        screen = capture_screen(SCREEN_REGION)
        x, y, _, _, _, _, confidence = find_template(screen, CHECK_FOR_ROOM3_END, threshold=0.5)
        if x is not None and y is not None:
            print(f"Room 2 end detected at: ({x}, {y}) with confidence {confidence:.3f}")
            break
        print("Room 2 end not detected. Performing double jump left and right...")

        # Double jump left followed by up press
        pyautogui.keyDown('left')
        pyautogui.press('c')  # First c press
        pyautogui.press('c')  # Second c press for double jump
        pyautogui.keyUp('left')
        time.sleep(0.2)  # Brief delay
        pyautogui.press('up')  # Press up after double jump
        time.sleep(0.5)  # Delay after action

        # Double jump right followed by up press
        pyautogui.keyDown('right')
        pyautogui.press('c')  # First c press
        pyautogui.press('c')  # Second c press for double jump
        pyautogui.keyUp('right')
        time.sleep(0.2)  # Brief delay
        pyautogui.press('up')  # Press up after double jump
        time.sleep(0.5)  # Delay after action

    print("Mission 2: Completed!")


def mission_3():
    print("Mission 3: Starting...")
    print("Mission 3: Performing tasks...")
    # Placeholder for Mission 3 tasks
    time.sleep(2)
    print("Mission 3: Completed!")

def mission_4():
    print("Mission 4: Starting...")
    print("Mission 4: Performing tasks...")
    # Placeholder for Mission 4 tasks
    time.sleep(2)
    print("Mission 4: Completed!")

def mission_5():
    print("Mission 5: Starting...")
    print("Mission 5: Performing tasks...")
    # Placeholder for Mission 5 tasks
    time.sleep(2)
    print("Mission 5: Completed!")

def mission_6():
    print("Mission 6: Starting...")
    print("Mission 6: Performing tasks...")
    # Placeholder for Mission 6 tasks
    time.sleep(2)
    print("Mission 6: Completed!")

def mission_7():
    print("Mission 7: Starting...")
    print("Mission 7: Performing tasks...")
    # Placeholder for Mission 7 tasks
    time.sleep(2)
    print("Mission 7: Completed!")

def main():
    print("Starting program... Press Ctrl+C to stop.")
    
    # Perform knight interaction once at the start
    knight_interaction()

    # Run all missions
    mission_1()
    mission_2()
    #mission_3()
    #mission_4()
    #mission_5()
    #mission_6()
    #mission_7()

    print("All missions completed!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Automation stopped.")