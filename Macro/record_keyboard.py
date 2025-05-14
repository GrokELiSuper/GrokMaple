#!/usr/bin/env python3
# keylogger_csv.py

import time
from pynput import keyboard

events = []
start_time = None

def normalize_key(key):
    """
    Return a plain string for the key:
    - for alphanumeric keys, key.char
    - for special keys (arrows, ctrl, etc.), key.name
    """
    try:
        # alphanumeric
        return key.char
    except AttributeError:
        # special key
        return key.name

def on_press(key):
    ts = time.time()
    delta = int(ts - start_time)
    key_str = normalize_key(key)
    print(f"{delta},press,{key_str}")
    events.append((delta, 'press', key_str))

def on_release(key):
    ts = time.time()
    delta = int(ts - start_time)
    key_str = normalize_key(key)
    print(f"{delta},release,{key_str}")
    events.append((delta, 'release', key_str))
    # stop on Esc
    if key == keyboard.Key.esc:
        return False

def main():
    global start_time
    print("Starting key logger. Press ESC to stop.\n")
    start_time = time.time()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # (optional) if you still want a final summary in the same CSV format:
    # print("\n=== Full Key Log ===")
    # for delta, evtype, key_str in events:
    #     print(f"{delta},{evtype},{key_str}")

if __name__ == "__main__":
    main()
