import mss
import win32gui

TARGET_TITLE = "Hold'em"  # part of your poker table title

hwnd = None
def enum_handler(h, ctx):
    global hwnd
    if win32gui.IsWindowVisible(h):
        title = win32gui.GetWindowText(h)
        if TARGET_TITLE in title:
            hwnd = h

win32gui.EnumWindows(enum_handler, None)

if not hwnd:
    raise SystemExit(f"No window found with title containing '{TARGET_TITLE}'")

rect = win32gui.GetWindowRect(hwnd)
print(f"Capturing {win32gui.GetWindowText(hwnd)} @ {rect}")

with mss.mss() as sct:
    img = sct.grab({
        "left": rect[0],
        "top": rect[1],
        "width": rect[2] - rect[0],
        "height": rect[3] - rect[1]
    })

import cv2, numpy as np
cv2.imwrite("calib_frame.png", np.array(img))
print("Saved calib_frame.png")
