import os, cv2, mss, win32gui, numpy as np

ASSETS_DIR = "assets/cards"
os.makedirs(ASSETS_DIR, exist_ok=True)
TEMPLATE_SIZE = (70, 100)  # saved template size (w,h)

# -------- window + monitor helpers --------
KEYWORDS = ["hold'em", "no limit", "money in play", "acr", "americas"]

def get_table_rect():
    cands = []
    def _enum(h,_):
        if not win32gui.IsWindowVisible(h): return
        t = win32gui.GetWindowText(h)
        if not t: return
        if any(k in t.lower() for k in KEYWORDS):
            L,T,R,B = win32gui.GetWindowRect(h)
            w,h = R-L, B-T
            if w>300 and h>300: cands.append((w*h,(L,T,R,B),t))
    win32gui.EnumWindows(_enum, None)
    if not cands: return None
    _, rect, title = max(cands, key=lambda x:x[0])
    print(f"[ok] Using table window: '{title}' {rect}")
    return rect

def pick_monitor_for_rect(rect, monitors):
    L,T,R,B = rect
    def overlap(a,b):
        L1,T1,R1,B1=a; L2,T2,R2,B2=b
        return max(0, min(R1,R2)-max(L1,L2)) * max(0, min(B1,B2)-max(T1,T2))
    best_i, best_a = None, -1
    for i,m in enumerate(monitors):
        if i==0: continue
        mr = (m['left'], m['top'], m['left']+m['width'], m['top']+m['height'])
        a = overlap(rect, mr)
        if a>best_a: best_a, best_i = a, i
    return best_i

def clamp_to_monitor(rect, mon):
    L,T,R,B = rect
    ml,mt = mon['left'], mon['top']
    mr,mb = ml+mon['width'], mt+mon['height']
    L2, T2 = max(L,ml), max(T,mt)
    R2, B2 = min(R,mr), min(B,mb)
    if R2<=L2 or B2<=T2: return None
    return (L2,T2,R2,B2)

def place_preview_outside(table, mon, pref_w=720, pref_h=420, pad=14):
    """Return a (x,y,w,h) for the preview that doesn't overlap the table rect on this monitor."""
    L,T,R,B = table
    ml,mt = mon['left'], mon['top']; mr,mb = ml+mon['width'], mt+mon['height']
    # Try right of table
    if R + pref_w + pad <= mr:
        return (R + pad, T, pref_w, pref_h)
    # Try left
    if L - pref_w - pad >= ml:
        return (L - pref_w - pad, T, pref_w, pref_h)
    # Try below
    if B + pref_h + pad <= mb:
        return (L, B + pad, pref_w, pref_h)
    # Try above
    if T - pref_h - pad >= mt:
        return (L, T - pref_h - pad, pref_w, pref_h)
    # Last resort: put it at monitor corner (may overlap a bit but smaller)
    return (ml + pad, mt + pad, min(pref_w, 600), min(pref_h, 360))

# -------- main --------
def main():
    table = get_table_rect()
    if not table:
        print("[err] Table window not found. Make sure it's visible.")
        return

    with mss.mss() as sct:
        mons = sct.monitors
        print("[info] Monitors:", mons)
        idx = pick_monitor_for_rect(table, mons)
        if not idx:
            print("[err] No overlapping monitor for the table.")
            return
        mon = mons[idx]
        safe = clamp_to_monitor(table, mon)
        if not safe:
            print("[err] Table outside selected monitor after clamping.")
            return
    Ls,Ts,Rs,Bs = safe
    W,H = Rs-Ls, Bs-Ts
    print(f"[info] Capturing rect {Ls,Ts,Rs,Bs} size={W}x{H} on monitor index {idx}")

    # Setup preview window OUTSIDE capture rect
    px,py,pw,ph = place_preview_outside((Ls,Ts,Rs,Bs), mon)
    cv2.namedWindow("Template Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Template Capture", pw, ph)
    cv2.moveWindow("Template Capture", px, py)

    print("""
Hotkeys:
  S  - select ROI (click & drag), then type label (e.g., Ah, Kd) in console to save
  P  - save full preview frame to template_frame.png
  ESC/Q - quit
Tip: keep this window OUTSIDE the table area to avoid the 'hall of mirrors'.
""")

    with mss.mss() as sct:
        while True:
            img = np.array(sct.grab({"left": Ls, "top": Ts, "width": W, "height": H}))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # draw table border so you see bounds
            cv2.rectangle(frame, (2,2), (W-2,H-2), (0,255,255), 2)
            cv2.imshow("Template Capture", frame)
            k = cv2.waitKey(20) & 0xFF

            if k in (27, ord('q')):
                break

            if k == ord('p'):
                cv2.imwrite("template_frame.png", frame)
                print("[ok] wrote template_frame.png")

            if k == ord('s'):
                # PAUSE live capture while selecting ROI to avoid recursion
                cv2.imshow("Template Capture", frame)
                r = cv2.selectROI("Template Capture", frame, fromCenter=False, showCrosshair=True)
                x,y,w,h = map(int, r)
                if w>0 and h>0:
                    crop = frame[y:y+h, x:x+w]
                    crop = cv2.resize(crop, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)
                    label = input("Enter label for this card (e.g., Ah, Kd): ").strip()
                    if label:
                        out = os.path.join(ASSETS_DIR, f"{label}.png")
                        cv2.imwrite(out, crop)
                        print(f"[ok] saved {out}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
