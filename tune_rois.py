# tune_rois.py
# Live tuner for card ROIs (overlay_rois.json) + per-slot glyph BASES (bases_glyph.json)
# deps: pip install opencv-python mss pywin32 numpy

import os, json, time
import cv2, mss, win32gui, win32api
import numpy as np

ROI_FILE   = "overlay_rois.json"
BASES_FILE = "bases_glyph.json"

SLOTS = ["hole1","hole2","board1","board2","board3","board4","board5"]

# Default BASES if no bases_glyph.json yet: (y1, y2, x1, x2) as FRACTIONS of the ROI
DEFAULT_BASES = {
    "hole1":  [0.10, 0.42, 0.12, 0.46],
    "hole2":  [0.10, 0.42, 0.12, 0.46],
    "board1": [0.09, 0.41, 0.10, 0.44],
    "board2": [0.09, 0.41, 0.11, 0.45],
    "board3": [0.09, 0.41, 0.12, 0.46],
    "board4": [0.09, 0.41, 0.12, 0.46],
    "board5": [0.09, 0.41, 0.12, 0.46],
}

# ---------- window helpers ----------
KEYWORDS = ["hold'em","no limit","money in play","acr","americas"]

def get_table_rect():
    cands=[]
    def _enum(h,_):
        if not win32gui.IsWindowVisible(h): return
        t = win32gui.GetWindowText(h)
        if not t: return
        if any(k in t.lower() for k in KEYWORDS):
            L,T,R,B = win32gui.GetWindowRect(h); w,h = R-L, B-T
            if w>300 and h>300: cands.append((w*h,(L,T,R,B),t))
    win32gui.EnumWindows(_enum, None)
    if not cands:
        raise SystemExit("[err] Poker table window not found. Make sure it's visible.")
    _, rect, title = max(cands, key=lambda x:x[0])
    print(f"[ok] Using window: '{title}' {rect}")
    return rect

def pick_monitor_for_rect(rect, monitors):
    L,T,R,B = rect
    def overlap(a,b):
        L1,T1,R1,B1=a; L2,T2,R2,B2=b
        return max(0, min(R1,R2)-max(L1,L2)) * max(0, min(B1,B2)-max(T1,T2))
    best_i, best_a = None, -1
    for i,m in enumerate(monitors):
        if i==0: continue
        mr = (m['left'],m['top'],m['left']+m['width'],m['top']+m['height'])
        a = overlap(rect, mr)
        if a>best_a: best_a, best_i = a, i
    return best_i

def clamp_to_monitor(rect, mon):
    L,T,R,B = rect
    ml,mt = mon['left'], mon['top']
    mr,mb = ml+mon['width'], mt+mon['height']
    L2,T2 = max(L,ml), max(T,mt)
    R2,B2 = min(R,mr), min(B,mb)
    if R2<=L2 or B2<=T2: return None
    return (L2,T2,R2,B2)

def place_preview_outside(table_rect, mon, pref_w=950, pref_h=620, pad=14):
    L,T,R,B = table_rect
    ml,mt = mon['left'], mon['top']
    mr,mb = ml+mon['width'], mt+mon['height']
    if R + pref_w + pad <= mr:   return (R + pad, T, pref_w, pref_h)
    if L - pref_w - pad >= ml:   return (L - pref_w - pad, T, pref_w, pref_h)
    if B + pref_h + pad <= mb:   return (L, B + pad, pref_w, pref_h)
    if T - pref_h - pad >= mt:   return (L, T - pref_h - pad, pref_w, pref_h)
    return (ml + pad, mt + pad, min(pref_w, 800), min(pref_h, 500))

# ---------- IO ----------
def load_rois():
    if not os.path.exists(ROI_FILE):
        raise SystemExit(f"[err] {ROI_FILE} not found.")
    with open(ROI_FILE, "r") as f:
        rois = json.load(f)
    for k in SLOTS:
        if k not in rois:
            rois[k] = [100,100,100,140]
    return rois

def save_rois(rois):
    with open(ROI_FILE, "w") as f:
        json.dump(rois, f, indent=2)
    print(f"[ok] saved {ROI_FILE}")

def load_bases():
    if os.path.exists(BASES_FILE):
        with open(BASES_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}
    for k,v in DEFAULT_BASES.items():
        if k not in data:
            data[k] = list(v)
    return data

def save_bases(bases):
    with open(BASES_FILE, "w") as f:
        json.dump(bases, f, indent=2)
    print(f"[ok] saved {BASES_FILE}")

# ---------- drawing ----------
def draw_slot(frame, name, box, glyph_base, highlight=False):
    x,y,w,h = map(int, box)
    color = (0,255,0) if highlight else (0,200,200)
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
    gy1,gy2,gx1,gx2 = glyph_base
    gy1 = int(y + gy1*h); gy2 = int(y + gy2*h)
    gx1 = int(x + gx1*w); gx2 = int(x + gx2*w)
    cv2.rectangle(frame, (gx1,gy1), (gx2,gy2), (0,165,255), 2)
    cv2.putText(frame, name, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def clamp_roi(box, W, H):
    x,y,w,h = box
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(10, min(w, W - x))
    h = max(10, min(h, H - y))
    return [int(x),int(y),int(w),int(h)]

# ---------- key handling ----------
# OpenCV arrow codes (waitKeyEx) and Windows VK codes (fallback)
ARROW_LEFT_CODES  = {2424832, 0x25}
ARROW_UP_CODES    = {2490368, 0x26}
ARROW_RIGHT_CODES = {2555904, 0x27}
ARROW_DOWN_CODES  = {2621440, 0x28}

def is_left(k):  return k in ARROW_LEFT_CODES
def is_right(k): return k in ARROW_RIGHT_CODES
def is_up(k):    return k in ARROW_UP_CODES
def is_down(k):  return k in ARROW_DOWN_CODES

HELP = """
Controls:
  1..7         select slot (1=hole1, 2=hole2, 3=board1, 4=board2, 5=board3, 6=board4, 7=board5)
  Arrows       move ROI (Shift = 10px step)
  Ctrl+Arrows  resize ROI (Shift = 10px step)
  Alt+Arrows   tweak glyph BASES for selected slot (±0.01 in fraction)
               Alt+Left/Right -> x1/x2   Alt+Up/Down -> y1/y2
  S            save overlay_rois.json + bases_glyph.json
  P            save screenshot
  K            toggle key-code debug print
  Q / ESC      quit
"""

def main():
    print(HELP)

    rois  = load_rois()
    bases = load_bases()

    rect = get_table_rect()
    with mss.mss() as sct:
        idx = pick_monitor_for_rect(rect, sct.monitors)
        if not idx:
            raise SystemExit("[err] No overlapping monitor.")
        mon = sct.monitors[idx]
        safe = clamp_to_monitor(rect, mon)
        if not safe:
            raise SystemExit("[err] Table outside selected monitor.")
        Ls,Ts,Rs,Bs = safe
        W,H = Rs-Ls, Bs-Ts

        px,py,pw,ph = place_preview_outside((Ls,Ts,Rs,Bs), mon)
        cv2.namedWindow("ROI Tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI Tuner", pw, ph)
        cv2.moveWindow("ROI Tuner", px, py)

        sel = "board3"  # default selection
        print(f"[info] selected: {sel}")

        debug_keys = False

        while True:
            frame = np.array(sct.grab({"left": Ls, "top": Ts, "width": W, "height": H}))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            for name in SLOTS:
                draw_slot(frame, name, rois[name], bases[name], highlight=(name==sel))

            cv2.putText(frame, "S: save   Ctrl+Arrows: resize   Alt+Arrows: tweak glyph   Shift: faster",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Selected: {sel}  ROI: {rois[sel]}  BASES: {np.round(bases[sel],3).tolist()}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

            cv2.imshow("ROI Tuner", frame)
            k = cv2.waitKeyEx(30)
            if k == -1:
                continue

            if k in (27, ord('q')):  # ESC/Q
                break

            if k in (ord('k'), ord('K')):
                debug_keys = not debug_keys
                print(f"[debug] key logging: {'ON' if debug_keys else 'OFF'}")
                continue

            if debug_keys:
                print(f"[key] {k}")

            if ord('1') <= k <= ord('7'):
                sel = SLOTS[k - ord('1')]
                print(f"[info] selected: {sel}")
                continue

            # modifiers
            step = 10 if (win32api.GetKeyState(0x10) < 0) else 2      # VK_SHIFT
            ctrl = (win32api.GetKeyState(0x11) < 0)                   # VK_CONTROL
            alt  = (win32api.GetKeyState(0x12) < 0)                   # VK_MENU

            x,y,w,h = rois[sel]
            y1,y2,x1,x2 = bases[sel]

            if not ctrl and not alt:
                # move ROI
                if is_left(k):  x -= step
                if is_right(k): x += step
                if is_up(k):    y -= step
                if is_down(k):  y += step

            elif ctrl and not alt:
                # resize ROI
                if is_left(k):  w -= step
                if is_right(k): w += step
                if is_up(k):    h -= step
                if is_down(k):  h += step

            elif alt and not ctrl:
                # tweak glyph bases (fractions)
                delta = 0.01 if (win32api.GetKeyState(0x10) >= 0) else 0.02  # Shift held → bigger steps
                if is_left(k):   x1 = max(0.00, x1 - delta)
                if is_right(k):  x2 = min(1.00, x2 + delta)
                if is_up(k):     y1 = max(0.00, y1 - delta)
                if is_down(k):   y2 = min(1.00, y2 + delta)
                bases[sel] = [float(y1), float(y2), float(x1), float(x2)]

            rois[sel] = clamp_roi([x,y,w,h], W, H)

            if k == ord('s') or k == ord('S'):
                save_rois(rois)
                save_bases(bases)
            if k == ord('p') or k == ord('P'):
                ts = int(time.time()*1000)
                cv2.imwrite(f"tuner_frame_{ts}.png", frame)
                print(f"[ok] wrote tuner_frame_{ts}.png")

        cv2.destroyAllWindows()
        save_rois(rois)
        save_bases(bases)

if __name__ == "__main__":
    main()
