import mss
import win32gui
import cv2, numpy as np

found_hwnd = None  # <- different name so we don't shadow it

def enum_handler(h, ctx):
    global found_hwnd
    if not win32gui.IsWindowVisible(h):
        return
    title = win32gui.GetWindowText(h)
    if "hold'em" in title.lower():           # case-insensitive
        print(f"HWND: {h}, Title: {title}, Rect: {win32gui.GetWindowRect(h)}")
        found_hwnd = h                        # <- save it (first match)

win32gui.EnumWindows(enum_handler, None)

if found_hwnd:
    L, T, R, B = win32gui.GetWindowRect(found_hwnd)
    with mss.mss() as sct:
        img = sct.grab({"left": L, "top": T, "width": R-L, "height": B-T})
        cv2.imwrite("table_test.png", np.array(img))
        print("Saved table_test.png")
else:
    print("Table window not found.")
