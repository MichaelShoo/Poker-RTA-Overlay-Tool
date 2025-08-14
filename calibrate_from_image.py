import json, cv2, os

IMG_PATH = os.path.abspath("calib_frame.png")
print("Opening:", IMG_PATH)
img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit("Couldn't open calib_frame.png. Run calibrate_rois.py first, or fix the working directory.")

print("Image shape:", img.shape, "mean brightness:", img.mean())

ROI_NAMES = ["hole1","hole2","board1","board2","board3","board4","board5","pot","tocall"]
rois = {}
ix=iy=wx=hy=0
drawing=False
current=0

disp = img.copy()

def redraw():
    disp[:] = img
    txt = f"{current+1}/{len(ROI_NAMES)}: {ROI_NAMES[current]}  — drag a box, press ENTER to accept (Q=quit)"
    cv2.rectangle(disp, (0,0), (disp.shape[1], 36), (32,32,32), -1)
    cv2.putText(disp, txt, (12,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    # show already-saved boxes
    for name, (x,y,w,h) in rois.items():
        cv2.rectangle(disp, (x,y), (x+w,y+h), (255,255,0), 2)
        cv2.putText(disp, name, (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

def on_mouse(event, x, y, flags, param):
    global ix, iy, wx, hy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True; ix,iy=x,y; wx,hy=x,y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        wx,hy=x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing=False; wx,hy=x,y
    redraw()
    if drawing or (ix!=wx and iy!=hy):
        x1,y1 = min(ix,wx), min(iy,hy)
        x2,y2 = max(ix,wx), max(iy,hy)
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.namedWindow("ROI Setup", cv2.WINDOW_NORMAL)
# scale to fit but keep it visible
scale = min(1200/img.shape[1], 800/img.shape[0], 1.0)
cv2.resizeWindow("ROI Setup", int(img.shape[1]*scale), int(img.shape[0]*scale))
cv2.moveWindow("ROI Setup", 60, 60)
cv2.setMouseCallback("ROI Setup", on_mouse)

redraw()
while True:
    cv2.imshow("ROI Setup", disp)
    k = cv2.waitKey(20) & 0xFFFF
    if k in (27, ord('q')):  # quit
        break
    if k in (13, 10):  # ENTER
        x1,y1 = min(ix,wx), min(iy,hy)
        x2,y2 = max(ix,wx), max(iy,hy)
        w,h = max(1, x2-x1), max(1, y2-y1)
        rois[ROI_NAMES[current]] = [int(x1), int(y1), int(w), int(h)]
        print("Saved", ROI_NAMES[current], rois[ROI_NAMES[current]])
        current += 1
        if current == len(ROI_NAMES):
            with open("overlay_rois.json","w") as f:
                json.dump(rois, f, indent=2)
            print("Wrote overlay_rois.json ✓")
            break
        ix=iy=wx=hy=0
        redraw()

cv2.destroyAllWindows()
