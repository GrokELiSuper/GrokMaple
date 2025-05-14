import win32gui
import pyautogui
import time
import random
import cv2
import numpy as np
from mss import mss
import os

# Configure pyautogui
pyautogui.FAILSAFE = True  # Move mouse to upper-left corner to stop the script
pyautogui.PAUSE = 0.05  # Small delay between actions

# Function to check if MapleStory Worlds is the active window
def is_maple_active():
    try:
        active_window = win32gui.GetWindowText(win32gui.GetForegroundWindow())
        return "MapleStory" in active_window  # Adjust based on your window title
    except:
        return False

# Function to perform two attacks in a given direction
def perform_attacks(direction):
    print(f"Attacking twice while facing {direction}...")
    # First attack
    pyautogui.keyDown("ctrl")
    time.sleep(random.uniform(0.2, 0.5))
    pyautogui.keyUp("ctrl")
    time.sleep(random.uniform(0.05, 0.1))  # Small pause between attacks
    # Second attack
    pyautogui.keyDown("ctrl")
    time.sleep(random.uniform(0.2, 0.5))
    pyautogui.keyUp("ctrl")

# Function to use a skill (alternating between 1 and 2, or use 3 if multiple enemies are close)
def use_skill(last_skill, enemies_within_200):
    if enemies_within_200 >= 2:  # If 2 or more enemies are within 200 pixels, use skill 3
        skill_key = "3"
        print("Multiple enemies within 200 pixels! Using high-MP skill on key 3...")
        pyautogui.press(skill_key)
        time.sleep(random.uniform(0.5, 1.0))  # Longer cooldown for skill 3
        return last_skill  # Don't change last_skill since we used 3
    else:
        skill_key = "1" if last_skill == "2" else "2"  # Alternate between 1 and 2
        print(f"Using magician skill on key {skill_key}...")
        pyautogui.press(skill_key)
        time.sleep(random.uniform(0.3, 0.7))  # Cooldown for skill
        return skill_key

# Function to mirror an image
def mirror_image(image):
    return cv2.flip(image, 1)  # Flip horizontally

# Function to detect the player and enemies
def detect_player_and_enemies():
    try:
        with mss() as sct:
            # Capture a region based on the game window size (1284x753)
            monitor = sct.monitors[1]  # Primary monitor
            # Extended capture region to 1000x600
            game_width, game_height = 1284, 753
            left = (monitor["width"] // 2) - (game_width // 2) + 142  # Center of game window - 500
            top = (monitor["height"] // 2) - (game_height // 2) + 76  # Center of game window - 300
            width = 1000
            height = 600
            region = {"left": left, "top": top, "width": width, "height": height}

            # Capture the screen
            screenshot = np.array(sct.grab(region))

        # Convert screenshot to grayscale
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Player templates
        player_templates = {
            "left": cv2.imread(os.path.join("images", "player_left_template.png"), cv2.IMREAD_GRAYSCALE),
            "right": cv2.imread(os.path.join("images", "player_right_template.png"), cv2.IMREAD_GRAYSCALE)
        }

        # Enemy templates (load and mirror each)
        enemy_template_files = [
            "blue_slime_template.png",
            "red_slime_template.png",
            "green_slime_template.png",  # Added for green slimes in your screenshot
            "mashroom_template.png",
            "mashroom_2_template.png",
            "pig_1_template.png",
            "pig_2_template.png"
        ]
        enemy_templates = []
        for template_file in enemy_template_files:
            template = cv2.imread(os.path.join("images", template_file), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                enemy_templates.append(template)  # Original
                enemy_templates.append(mirror_image(template))  # Mirrored

        # Check if templates loaded correctly
        for direction, template in player_templates.items():
            if template is None:
                print(f"Error: Could not load player_{direction}_template.png.")
                return None, None, None, 0
        if not enemy_templates:
            print("Error: Could not load any enemy templates.")
            return None, None, None, 0

        # Detect player
        player_position = None
        player_direction = None
        player_confidence = 0
        for direction, template in player_templates.items():
            result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            print(f"Player {direction} detection confidence: {max_val}")
            if max_val >= 0.4 and max_val > player_confidence:  # Lowered threshold
                player_position = max_loc
                player_direction = direction
                player_confidence = max_val

        # Detect enemies (track multiple enemies)
        enemy_positions = []
        enemies_within_200 = 0
        for i, template in enumerate(enemy_templates):
            result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.4  # Lowered threshold
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                enemy_positions.append(pt)
                # Check if this enemy is within 200 pixels of the player
                if player_position is not None:
                    distance = abs(pt[0] - player_position[0])
                    if distance <= 200:
                        enemies_within_200 += 1

        # Select the closest enemy
        closest_enemy_position = None
        if enemy_positions:
            if player_position is not None:
                # Find the closest enemy based on x-coordinate
                closest_enemy_position = min(enemy_positions, key=lambda pos: abs(pos[0] - player_position[0]))
            else:
                # If player not detected, just pick the first enemy
                closest_enemy_position = enemy_positions[0]

        return player_position, player_direction, closest_enemy_position, enemies_within_200

    except Exception as e:
        print(f"Error in detection: {e}")
        return None, None, None, 0

# Function to simulate a skill-focused routine with enemy detection
def maple_macro():
    print("Starting MapleStory Worlds macro (skill-focused with enemy detection)...")
    print("Press Ctrl+C to stop the macro, or move mouse to upper-left corner.")
    time.sleep(3)  # Give time to switch to the game

    last_skill = "2"  # Start with skill 2 to alternate
    last_auto_attack_time = time.time()  # Track the last time an auto-attack was used

    try:
        while True:
            # Only proceed if MapleStory Worlds is active
            if not is_maple_active():
                print("MapleStory Worlds is not active. Pausing...")
                time.sleep(1)
                continue

            # Detect player and enemies
            player_position, player_direction, enemy_position, enemies_within_200 = detect_player_and_enemies()

            if player_position is None:
                print("Player not detected. Assuming player is in the center and proceeding...")
                # Fallback: assume player is in the center of the capture region
                player_position = (500, 300)  # Center of 1000x600 region
                player_direction = random.choice(["left", "right"])
            else:
                print(f"Player detected at position: {player_position}, facing: {player_direction}")

            if enemy_position is None:
                print("No enemy detected. Waiting for enemies...")
                time.sleep(0.5)  # Wait instead of moving randomly
            else:
                # Enemy detected: determine relative position
                player_x = player_position[0]
                enemy_x = enemy_position[0]
                print(f"Player at x={player_x}, facing {player_direction}. Enemy at x={enemy_x}")

                # Decide which direction to face
                if enemy_x < player_x:
                    direction = "left"  # Enemy is to the left
                else:
                    direction = "right"  # Enemy is to the right

                # Turn to face the enemy if not already facing that direction
                if player_direction != direction:
                    print(f"Turning to face {direction}...")
                    pyautogui.keyDown(direction)
                    time.sleep(0.2)  # Small movement to turn
                    pyautogui.keyUp(direction)
                    player_direction = direction  # Update the direction

                # Calculate distance to the enemy
                distance = abs(enemy_x - player_x)
                print(f"Distance to enemy: {distance} pixels")

                # Move if the enemy is farther than 400 pixels
                if distance > 400:
                    print(f"Enemy is far (>400 pixels). Moving {direction} for 1 second...")
                    pyautogui.keyDown(direction)
                    time.sleep(1)  # Move for 1 second
                    pyautogui.keyUp(direction)

                # Use a skill (always use skills when enemy is detected)
                last_skill = use_skill(last_skill, enemies_within_200)

                # Check if we should auto-attack (every 10 seconds or if enemy is very close)
                current_time = time.time()
                time_since_last_auto_attack = current_time - last_auto_attack_time
                if (time_since_last_auto_attack >= 10) or (distance <= 50):  # Every 10 seconds or enemy within 50 pixels
                    perform_attacks(direction)
                    last_auto_attack_time = current_time  # Reset the timer

            # Pick up items frequently (90% chance, assuming Z is the key)
            if random.random() > 0.1:
                print("Picking up items...")
                pyautogui.press("z")
                time.sleep(0.1)

            # Jump (20% chance, to look slightly more natural)
            if random.random() > 0.8:
                print("Jumping...")
                pyautogui.keyDown("alt")
                time.sleep(0.1)
                pyautogui.keyUp("alt")

            # Use a potion if needed (30% chance, assuming F1 is a potion hotkey)
            if random.random() > 0.7:
                print("Using a potion...")
                pyautogui.press("f1")
                time.sleep(0.1)

            # Short pause between cycles to avoid looking too robotic
            time.sleep(random.uniform(0.05, 0.2))

            # Occasional longer pause (5% chance, to mimic a player thinking)
            if random.random() > 0.95:
                print("Taking a short break...")
                time.sleep(random.uniform(1, 3))

    except KeyboardInterrupt:
        print("Macro stopped by user.")

if __name__ == "__main__":
    maple_macro()