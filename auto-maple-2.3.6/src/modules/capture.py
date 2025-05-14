"""A module for tracking useful in-game information."""

import time
import cv2
import threading
import ctypes
import mss
import mss.windows
import numpy as np
from src.common import config, utils
from ctypes import wintypes
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

# Hardcoded minimap coordinates (based on 1366x768 resolution)
MINIMAP_TOP_LEFT = (5, 120)  # Top-left corner of the map area (below title bar)
MINIMAP_BOTTOM_RIGHT = (344, 288)  # Bottom-right corner of the map area

# Offset in pixels to adjust for windowed mode (set to 0 for full-screen)
WINDOWED_OFFSET_TOP = 0
WINDOWED_OFFSET_LEFT = 0

# The player's symbol on the minimap
PLAYER_TEMPLATE = cv2.imread('assets/player_template.png', 0)
PT_HEIGHT, PT_WIDTH = PLAYER_TEMPLATE.shape

class Capture:
    """
    A class that tracks player position and various in-game events. It constantly updates
    the config module with information regarding these events. It also annotates and
    displays the minimap in a pop-up window.
    """

    def __init__(self):
        """Initializes this Capture object's main thread."""

        config.capture = self

        self.frame = None
        self.minimap = {}
        self.minimap_ratio = 1
        self.minimap_sample = None
        self.sct = None
        self.window = {
            'left': 0,
            'top': 0,
            'width': 1366,
            'height': 768
        }
        self.map_offset = (0, 0)  # Track the map's offset relative to the player

        self.ready = False
        self.calibrated = False
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

    def start(self):
        """Starts this Capture's thread."""

        print('\n[~] Started video capture')
        self.thread.start()

    def _main(self):
        """Constantly monitors the player's position and in-game events."""

        mss.windows.CAPTUREBLT = 0
        while True:
            # Try to find the game window
            handle = user32.FindWindowW(None, 'MapleStory WORLD-MAPLE NUNU WORLD')
            if handle != 0:
                print('\n[~] Found Nunu World window')
                rect = wintypes.RECT()
                user32.GetWindowRect(handle, ctypes.pointer(rect))
                rect = (rect.left, rect.top, rect.right, rect.bottom)
                rect = tuple(max(0, x) for x in rect)

                self.window['left'] = rect[0] + WINDOWED_OFFSET_LEFT
                self.window['top'] = rect[1] + WINDOWED_OFFSET_TOP
                self.window['width'] = max(rect[2] - rect[0] - WINDOWED_OFFSET_LEFT * 2, 1)
                self.window['height'] = max(rect[3] - rect[1] - WINDOWED_OFFSET_TOP, 1)
            else:
                # Fallback to capturing the main monitor (assuming full-screen on main monitor)
                print('\n[~] Could not find Nunu World window. Capturing main monitor instead.')
                with mss.mss() as sct:
                    monitor = sct.monitors[1]  # Main monitor (index 1)
                    self.window['left'] = monitor["left"]
                    self.window['top'] = monitor["top"]
                    self.window['width'] = monitor["width"]
                    self.window['height'] = monitor["height"]

            # Calibrate by directly cropping the minimap region
            with mss.mss() as self.sct:
                self.frame = self.screenshot()
            if self.frame is None:
                continue
            mm_tl = MINIMAP_TOP_LEFT
            mm_br = MINIMAP_BOTTOM_RIGHT
            print(f"Minimap region: Top-Left {mm_tl}, Bottom-Right {mm_br}")
            self.minimap_ratio = (mm_br[0] - mm_tl[0]) / (mm_br[1] - mm_tl[1])
            self.minimap_sample = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]
            self.calibrated = True

            with mss.mss() as self.sct:
                while True:
                    if not self.calibrated:
                        break

                    # Take screenshot
                    self.frame = self.screenshot()
                    if self.frame is None:
                        continue

                    # Crop the frame to only show the minimap
                    minimap = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]

                    # Determine the player's position
                    player = utils.multi_match(minimap, PLAYER_TEMPLATE, threshold=0.8)
                    if player:
                        # Player position relative to the minimap center
                        player_pos = utils.convert_to_relative(player[0], minimap)
                        # Estimate map offset (assuming player is near the center)
                        center_pos = (0.5, 0.5)
                        self.map_offset = (
                            player_pos[0] - center_pos[0],
                            player_pos[1] - center_pos[1]
                        )
                        config.player_pos = player_pos
                        print(f"Player position: {config.player_pos}, Map offset: {self.map_offset}")

                    # Package display information to be polled by GUI
                    self.minimap = {
                        'minimap': minimap,
                        'rune_active': config.bot.rune_active,
                        'rune_pos': config.bot.rune_pos,
                        'path': config.path,
                        'player_pos': config.player_pos
                    }

                    if not self.ready:
                        self.ready = True
                    time.sleep(0.001)

    def screenshot(self, delay=1):
        try:
            return np.array(self.sct.grab(self.window))
        except mss.exception.ScreenShotError:
            print(f'\n[!] Error while taking screenshot, retrying in {delay} second'
                  + ('s' if delay != 1 else ''))
            time.sleep(delay)