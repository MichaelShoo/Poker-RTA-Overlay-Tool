# test_recognition.py
# Robust ACR reader for hole cards + board
# - Presence gate
# - Hole cards: GLYPH-ONLY + exhaustive corner scan (position/scale sweep)
# - Board: glyph-first with full-card fallback
# - Auto-align (slot-aware), cyan face box, amber dashed fallback
# - Ten-aware ranks with 8 vs 10 disambiguation
#
# deps: pip install opencv-python mss pywin32 numpy

import os, json, time
import cv2, mss, win32gui
import numpy as np
from glob import glob

# ===== Paths / knobs =====
TEMPLATE_DIR = os.path.join("assets", "cards")
ROI_JSON     = "overlay_rois.json"

# Presence thresholds
P_WHITE_MIN   = {"hole1":0.13,"hole2":0.13,"board1":0.14,"board2":0.14,"board3":0.14,"board4":0.14,"board5":0.14}
P_VAL_MIN, P_GREEN_MAX, P_BACK_RED, P_WHITE_BACK_MAX = 145, 0.55, 0.20, 0.08

# Glyph thresholds
GLYPH_STRICT, GLYPH_ACCEPT, GLYPH_MIN = 0.82, 0.80, 0.66

# Full-card threshold (board only)
FULLCARD_ACCEPT = 0.80

# Red/black gating
RED_CUTOFF = 0.08
COLOR_GATING_ENABLED = True  # toggle with 'G'

# Neighborhood search around each per-slot glyph box (fractions of *card face crop*)
DELTA_STEPS = [-0.06, -0.03, 0.0, 0.03, 0.06]

CARD_ROI_KEYS = ["hole1","hole2","board1","board2","board3","board4","board5"]
RANKS = list("23456789TJQKA")
SUITS = list("cdhs")
RED_SUITS, BLACK_SUITS = {"h","d"}, {"s","c"}

# Glyph base boxes relative to the *card face crop*
BASES = {
    "hole1":(0.10,0.42,0.12,0.46), "hole2":(0.10,0.42,0.12,0.46),
    "board1":(0.09,0.41,0.10,0.44), "board2":(0.09,0.41,0.11,0.45),
    "board3":(0.09,0.41,0.12,0.46), "board4":(0.09,0.41,0.12,0.46), "board5":(0.09,0.41,0.12,0.46),
}

# Rank crop base (non-scan)
RANK_X_FRAC = 0.62

# --- Face detector tuning (looser) ---
FACE_WHITE_S_MAX, FACE_WHITE_V_MIN = 120, 160
FACE_MIN_AREA_FRAC, FACE_RATIO_MIN, FACE_RATIO_MAX = 0.04, 0.45, 1.10

# Hole-card corner scan windows (fractions of face crop)
SCAN_SCALES   = (0.85, 0.92, 1.00, 1.08, 1.16)
RANK_SCAN_BOX = (0.02, 0.50, 0.02, 0.58)        # higher & wider for rank
SUIT_SCAN_BOX = (0.18, 0.48, 0.04, 0.44)        # tighter to mini pip to avoid big pip

# ===== Window helpers =====
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
    if not cands: raise SystemExit("[err] Poker table window not found (keywords).")
    _, rect, title = max(cands, key=lambda x:x[0])
    print(f"[ok] Using window: '{title}' {rect}")
    return rect

def pick_monitor_for_rect(rect, monitors):
    L,T,R,B = rect
    def overlap(a,b):
        L1,T1,R1,B1=a; L2,T2,R2,B2=b
        return max(0, min(R1,R2)-max(L1,L2))*max(0, min(B1,B2)-max(T1,T2))
    best_i, best_a = None, -1
    for i,m in enumerate(monitors):
        if i==0: continue
        mr = (m['left'],m['top'],m['left']+m['width'],m['top']+m['height'])
        a = overlap(rect, mr)
        if a>best_a: best_a, best_i = a, i
    return best_i

def clamp_to_monitor(rect, mon):
    L,T,R,B = rect
    ml,mt = mon['left'], mon['top']; mr,mb = ml+mon['width'], mt+mon['height']
    L2,T2 = max(L,ml), max(T,mt); R2,B2 = min(R,mr), min(B,mb)
    if R2<=L2 or B2<=T2: return None
    return (L2,T2,R2,B2)

def place_preview_outside(table_rect, mon, pref_w=900, pref_h=520, pad=14):
    L,T,R,B = table_rect; ml,mt = mon['left'],mon['top']; mr,mb = ml+mon['width'], mt+mon['height']
    if R + pref_w + pad <= mr:   return (R + pad, T, pref_w, pref_h)
    if L - pref_w - pad >= ml:   return (L - pref_w - pad, T, pref_w, pref_h)
    if B + pref_h + pad <= mb:   return (L, B + pad, pref_w, pref_h)
    if T - pref_h - pad >= mt:   return (L, T - pref_h - pad, pref_w, pref_h)
    return (ml + pad, mt + pad, min(pref_w, 700), min(pref_h, 420))

# ===== Template building =====
def load_full_templates(folder=TEMPLATE_DIR):
    tm_gray={}
    for p in glob(os.path.join(folder,"*.png")):
        lab = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g,(3,3),0)
        tm_gray[lab]=g
    if not tm_gray: raise SystemExit(f"[err] No templates found in {folder}.")
    print(f"[ok] Loaded {len(tm_gray)} full templates")
    return tm_gray

def _clahe(): return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

def build_rank_suit_templates(full_gray):
    """
    rank_multi: dict[label] -> list[np.ndarray]  (may contain 2 templates for Ten)
    rank_edge_multi: same but edge images
    suit_tm: dict[suit] -> np.ndarray (avg)
    """
    rank_multi={r:[] for r in RANKS}; suit_patches={s:[] for s in SUITS}
    clahe=_clahe()
    for label,g in full_gray.items():
        r,s=label[0],label[1]; H,W=g.shape[:2]
        y1,y2=int(0.10*H),int(0.42*H); x1,x2=int(0.10*W),int(0.45*W)
        patch=g[y1:y2,x1:x2]
        if patch.size==0: continue
        patch=cv2.resize(patch,(56,56),interpolation=cv2.INTER_AREA); patch=clahe.apply(patch)
        rank_left = patch[:28, :int(56*RANK_X_FRAC)]
        suit_full = patch[28:, :]
        if r in rank_multi: rank_multi[r].append(rank_left)    # left first
        if r=="T":  # add zero-half for 10
            zero_half = patch[:28, int(56*0.36):]
            rank_multi[r].append(zero_half)
        suit_patches[s].append(suit_full)
    suit_tm={}
    for s,arrs in suit_patches.items():
        if not arrs: continue
        acc=np.zeros_like(arrs[0],dtype=np.float32)
        for a in arrs: acc+=a.astype(np.float32)
        mean=(acc/max(1,len(arrs))).astype(np.uint8)
        suit_tm[s]=_clahe().apply(mean)
    rank_edge_multi={k:[cv2.Canny(v,80,160) for v in vs] for k,vs in rank_multi.items()}
    print(f"[ok] Built rank templates (multi): {{k:len(v) for k,v in rank_multi.items()}}")
    print(f"[ok] Built suit templates: {sorted(suit_tm.keys())}")
    return rank_multi, rank_edge_multi, suit_tm

# ===== Color helpers =====
def hsv(img_bgr): return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
def red_ratio(bgr):
    H=hsv(bgr); m1=cv2.inRange(H,(0,70,60),(10,255,255)); m2=cv2.inRange(H,(170,70,60),(180,255,255))
    return float(np.count_nonzero(cv2.bitwise_or(m1,m2)))/float(H.shape[0]*H.shape[1])
def green_ratio(bgr):
    H=hsv(bgr); mask=cv2.inRange(H,(35,40,40),(95,255,255))
    return float(np.count_nonzero(mask))/float(mask.size)
def white_ratio(bgr):
    H=hsv(bgr); mask=cv2.inRange(H,(0,0,185),(180,60,255))
    return float(np.count_nonzero(mask))/float(mask.size)
def mean_val(bgr): return float(hsv(bgr)[:,:,2].mean())

# ===== Presence gate =====
def card_present(roi_bgr, slot):
    if roi_bgr is None or roi_bgr.size==0: return False
    wr,gr,rr,v = white_ratio(roi_bgr), green_ratio(roi_bgr), red_ratio(roi_bgr), mean_val(roi_bgr)
    if rr>=P_BACK_RED and wr<=P_WHITE_BACK_MAX: return False
    if gr>=P_GREEN_MAX and wr<0.10: return False
    return (wr>=P_WHITE_MIN.get(slot,0.12)) and (v>=P_VAL_MIN)

# ===== Matching helpers =====
def match_top2_multi(gray, templates_multi, allowed=None):
    label_scores={}
    for k,tmps in templates_multi.items():
        if allowed is not None and k not in allowed: continue
        best=-1.0
        for t in tmps:
            if gray.shape[0]<t.shape[0] or gray.shape[1]<t.shape[1]: continue
            v=float(cv2.matchTemplate(gray,t,cv2.TM_CCOEFF_NORMED).max())
            if v>best: best=v
        if best>=0: label_scores[k]=best
    if not label_scores: return None,-1.0,-1.0
    sl=sorted(label_scores.items(), key=lambda x:x[1], reverse=True)
    best_label,best_v=sl[0]; second_v=sl[1][1] if len(sl)>1 else -1.0
    return best_label,best_v,second_v

def fullcard_match(roi_bgr, full_gray, allowed_suits=None):
    if roi_bgr is None or roi_bgr.size==0: return None,0.0
    roi_g=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY); roi_g=cv2.GaussianBlur(roi_g,(3,3),0)
    best_label,best_score=None,-1.0
    for scale in (0.75,0.80,0.90,1.00,1.10,1.20,1.30,1.40):
        resized=cv2.resize(roi_g,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
        rh,rw=resized.shape[:2]
        for label,tmpl in full_gray.items():
            if allowed_suits and label[1] not in allowed_suits: continue
            th,tw=tmpl.shape[:2]
            if rh<th or rw<tw: continue
            score=float(cv2.matchTemplate(resized,tmpl,cv2.TM_CCOEFF_NORMED).max())
            if score>best_score: best_score,best_label=score,label
    return best_label,best_score

def _prep_glyph(gray_56):
    g=_clahe().apply(gray_56); g=cv2.GaussianBlur(g,(3,3),0); return g

# ===== Auto-align utilities =====
def detect_card_face_candidates(roi_bgr):
    H,W=roi_bgr.shape[:2]
    hsv_img=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2HSV)
    m_white=cv2.inRange(hsv_img,(0,0,FACE_WHITE_V_MIN),(180,FACE_WHITE_S_MAX,255))
    m_gray =cv2.inRange(hsv_img,(0,0,140),(180,140,255))
    mask=cv2.bitwise_or(m_white,m_gray)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=1)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        g=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY); g=cv2.GaussianBlur(g,(3,3),0)
        th=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,-10)
        th=cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)
        cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return []
    cands=[]; min_area=FACE_MIN_AREA_FRAC*(W*H)
    for c in cnts:
        a=cv2.contourArea(c); 
        if a<min_area: continue
        x,y,w,h=cv2.boundingRect(c); ratio=w/float(h) if h>0 else 0.0
        if FACE_RATIO_MIN<=ratio<=FACE_RATIO_MAX: cands.append((x,y,w,h))
    return cands

def choose_face_box_for_slot(cands, slot):
    if not cands: return None
    if slot=="hole1": return sorted(cands,key=lambda b:b[0])[0]      # leftmost
    if slot=="hole2": return sorted(cands,key=lambda b:b[0])[-1]     # rightmost
    cands=sorted(cands,key=lambda b:b[2]*b[3],reverse=True)
    best_area=cands[0][2]*cands[0][3]
    tied=[b for b in cands if abs(b[2]*b[3]-best_area)<0.05*best_area]
    return sorted(tied,key=lambda b:b[0])[0]

def detect_card_face_box(roi_bgr, slot):
    cands=detect_card_face_candidates(roi_bgr); return choose_face_box_for_slot(cands,slot)

def card_face_crop(roi_bgr, slot):
    box=detect_card_face_box(roi_bgr,slot)
    if box is None: return roi_bgr
    x,y,w,h=box; crop=roi_bgr[y:y+h,x:x+w]; return crop if crop.size else roi_bgr

# ===== Hole-card corner scan helpers =====
def score_specific_label(region_gray, tmpl_list, scales):
    best=-1.0
    for t in tmpl_list:
        th,tw=t.shape[:2]
        for sc in scales:
            tw2,th2=int(tw*sc),int(th*sc)
            if tw2<6 or th2<6: continue
            t2=cv2.resize(t,(tw2,th2),interpolation=cv2.INTER_LINEAR)
            if region_gray.shape[0]<th2 or region_gray.shape[1]<tw2: continue
            v=float(cv2.matchTemplate(region_gray,t2,cv2.TM_CCOEFF_NORMED).max())
            if v>best: best=v
    return best

def glyph_exhaustive_scan(face_bgr, rank_multi, suit_tm, allowed_suits=None):
    if face_bgr is None or face_bgr.size==0: return None,0.0,0.0,0.0
    H,W=face_bgr.shape[:2]
    g=cv2.cvtColor(face_bgr,cv2.COLOR_BGR2GRAY); g=_clahe().apply(g); g=cv2.GaussianBlur(g,(3,3),0)
    # rank region
    ry1,ry2=int(RANK_SCAN_BOX[0]*H),int(RANK_SCAN_BOX[1]*H)
    rx1,rx2=int(RANK_SCAN_BOX[2]*W),int(RANK_SCAN_BOX[3]*W)
    rank_reg=g[ry1:ry2,rx1:rx2]
    # suit region
    sy1,sy2=int(SUIT_SCAN_BOX[0]*H),int(SUIT_SCAN_BOX[1]*H)
    sx1,sx2=int(SUIT_SCAN_BOX[2]*W),int(SUIT_SCAN_BOX[3]*W)
    suit_reg=g[sy1:sy2,sx1:sx2]

    # best rank among all
    r_key, r_v, r_v2 = match_top2_multi(rank_reg, rank_multi)

    # ---- Ten vs Eight disambiguation ----
    left = rank_reg[:, :max(2,int(0.20*rank_reg.shape[1]))]
    left_dark_ratio = float(np.count_nonzero(left < 160)) / float(left.size) if left.size else 0.0
    v8 = score_specific_label(rank_reg, rank_multi.get("8", []), SCAN_SCALES)
    vT_zero = score_specific_label(rank_reg, rank_multi.get("T", [])[1:2], SCAN_SCALES)  # zero-only template

    # If T wins due to zero-only, but left shows a stroke and 8 is close, flip to 8
    if r_key=="T" and vT_zero>=r_v-1e-6 and left_dark_ratio>0.06 and v8>=r_v-0.04:
        r_key, r_v = "8", v8

    # suit: restrict by color if needed
    suit_dict = suit_tm if (allowed_suits is None) else {k:v for k,v in suit_tm.items() if k in allowed_suits}
    s_best,s_best_v,s_second_v = None,-1.0,-1.0
    for k,t in suit_dict.items():
        for sc in SCAN_SCALES:
            th,tw=t.shape[:2]; tw2,th2=int(tw*sc),int(th*sc)
            if tw2<6 or th2<6: continue
            t2=cv2.resize(t,(tw2,th2),interpolation=cv2.INTER_LINEAR)
            if suit_reg.shape[0]<th2 or suit_reg.shape[1]<tw2: continue
            v=float(cv2.matchTemplate(suit_reg,t2,cv2.TM_CCOEFF_NORMED).max())
            if v>s_best_v: s_second_v=s_best_v; s_best_v=v; s_best=k
            elif v>s_second_v: s_second_v=v

    label = (f"{r_key}{s_best}" if r_key and s_best else None)
    gap_r = r_v - (r_v2 if r_v2>=0 else 0.0)
    gap_s = s_best_v - (s_second_v if s_second_v>=0 else 0.0)
    score = min(r_v, s_best_v)
    return label, score, gap_r, gap_s

# ===== Recognition =====
def recognize_card(roi_bgr, rank_multi, rank_edge_multi, suit_tm, full_gray, slot_name=None, glyph_only=False, face_bgr=None):
    if roi_bgr is None or roi_bgr.size==0: return None,0.0

    face_bgr = face_bgr if face_bgr is not None else card_face_crop(roi_bgr, slot_name)
    H,W=face_bgr.shape[:2]
    base=BASES.get(slot_name,(0.10,0.42,0.10,0.45))

    glyph_best_label,glyph_best_score = None,-1.0
    glyph_best_rank_gap,glyph_best_suit_gap = 0.0,0.0
    allowed_from_color=None

    probe=face_bgr[int(0.10*H):int(0.42*H), int(0.10*W):int(0.45*W)]
    rr_probe=red_ratio(probe) if probe.size else 0.0
    allowed_suits = RED_SUITS if (COLOR_GATING_ENABLED and rr_probe>RED_CUTOFF) else (BLACK_SUITS if COLOR_GATING_ENABLED else None)
    allowed_from_color = allowed_suits

    # A) Exhaustive scan for HOLE cards
    if glyph_only:
        label,score,gap_r,gap_s = glyph_exhaustive_scan(face_bgr, rank_multi, suit_tm, allowed_suits)
        if label: return label,score
        # otherwise fall through to local scan

    # B) Local neighborhood (fast)
    for dy in DELTA_STEPS:
        for dx in DELTA_STEPS:
            y1=int(max(0,(base[0]+dy)*H)); y2=int(min(H,(base[1]+dy)*H))
            x1=int(max(0,(base[2]+dx)*W)); x2=int(min(W,(base[3]+dx)*W))
            if y2<=y1 or x2<=x1: continue
            glyph_bgr=face_bgr[y1:y2, x1:x2]
            if glyph_bgr.size==0: continue
            glyph=cv2.cvtColor(glyph_bgr,cv2.COLOR_BGR2GRAY); glyph=cv2.resize(glyph,(56,56),interpolation=cv2.INTER_AREA); glyph=_prep_glyph(glyph)

            rx2=int(56*RANK_X_FRAC)
            rankA_g=glyph[:24,:rx2]; rankB_g=glyph[:28,:rx2]
            rankA_e=cv2.Canny(rankA_g,80,160); rankB_e=cv2.Canny(rankB_g,80,160)

            rA_g,rvA_g,rv2A_g=match_top2_multi(rankA_g,rank_multi)
            rB_g,rvB_g,rv2B_g=match_top2_multi(rankB_g,rank_multi)
            rA_e,rvA_e,rv2A_e=match_top2_multi(rankA_e,rank_edge_multi)
            rB_e,rvB_e,rv2B_e=match_top2_multi(rankB_e,rank_edge_multi)

            if rvA_e>rvA_g: rA,rvA,rv2A=rA_e,rvA_e,rv2A_e
            else:           rA,rvA,rv2A=rA_g,rvA_g,rv2A_g
            if rvB_e>rvB_g: rB,rvB,rv2B=rB_e,rvB_e,rv2B_e
            else:           rB,rvB,rv2B=rB_g,rvB_g,rv2B_g

            suitA=glyph[32:,:]; suitB=glyph[28:,:]
            # suit_tm is single-template per suit; adapt to multi interface
            sA,svA,sv2A=match_top2_multi(suitA,{k:[v] for k,v in suit_tm.items()},allowed=allowed_suits)
            sB,svB,sv2B=match_top2_multi(suitB,{k:[v] for k,v in suit_tm.items()},allowed=allowed_suits)

            scoreA=min(rvA,svA); scoreB=min(rvB,svB)
            gapA=(rvA-(rv2A if rv2A>=0 else 0.0))+(svA-(sv2A if sv2A>=0 else 0.0))
            gapB=(rvB-(rv2B if rv2B>=0 else 0.0))+(svB-(sv2B if sv2B>=0 else 0.0))

            if (scoreA,gapA)>=(glyph_best_score,glyph_best_rank_gap+glyph_best_suit_gap):
                glyph_best_score=scoreA; glyph_best_label=(f"{rA}{sA}" if rA and sA else None)
                glyph_best_rank_gap=rvA-(rv2A if rv2A>=0 else 0.0); glyph_best_suit_gap=svA-(sv2A if sv2A>=0 else 0.0)
            if (scoreB,gapB)>=(glyph_best_score,glyph_best_rank_gap+glyph_best_suit_gap):
                glyph_best_score=scoreB; glyph_best_label=(f"{rB}{sB}" if rB and sB else None)
                glyph_best_rank_gap=rvB-(rv2B if rv2B>=0 else 0.0); glyph_best_suit_gap=svB-(sv2B if sv2B>=0 else 0.0)

    if glyph_only: return glyph_best_label, glyph_best_score

    # Board: allow full-card fallback
    if glyph_best_label and glyph_best_score>=GLYPH_STRICT: return glyph_best_label,glyph_best_score
    need_fb=(glyph_best_score<GLYPH_MIN) or (glyph_best_rank_gap<0.08) or (glyph_best_suit_gap<0.08)
    if need_fb:
        fb_label,fb_score=fullcard_match(face_bgr,full_gray,allowed_suits=allowed_from_color)
        if fb_label and fb_score>=FULLCARD_ACCEPT: return fb_label,fb_score
        if glyph_best_label and glyph_best_score>=GLYPH_ACCEPT: return glyph_best_label,glyph_best_score
        if fb_label: return fb_label,fb_score
        return glyph_best_label,glyph_best_score
    if glyph_best_label and glyph_best_score>=GLYPH_ACCEPT: return glyph_best_label,glyph_best_score
    fb_label,fb_score=fullcard_match(face_bgr,full_gray,allowed_suits=allowed_from_color)
    if fb_label and fb_score>=FULLCARD_ACCEPT: return fb_label,fb_score
    return (glyph_best_label or fb_label), max(glyph_best_score,fb_score)

# ===== Small utils =====
def crop(img, box): x,y,w,h=map(int,box); return img[y:y+h, x:x+w]
def clamp_box_to_wh(box,W,H):
    x,y,w,h=map(int,box); x=max(0,min(x,W-1)); y=max(0,min(y,H-1))
    w=max(1,min(w,W-x)); h=max(1,min(h,H-y)); return [x,y,w,h]

# ===== Main =====
def main():
    global COLOR_GATING_ENABLED
    if not os.path.exists(ROI_JSON): raise SystemExit(f"[err] {ROI_JSON} not found. Run calibration first.")
    with open(ROI_JSON,"r") as f: ROIS=json.load(f)
    for k in CARD_ROI_KEYS:
        if k not in ROIS: raise SystemExit(f"[err] ROI '{k}' missing in {ROI_JSON}")

    full_gray = load_full_templates(TEMPLATE_DIR)
    rank_multi, rank_edge_multi, suit_tm = build_rank_suit_templates(full_gray)

    rect=get_table_rect()
    with mss.mss() as sct:
        idx=pick_monitor_for_rect(rect,sct.monitors)
        if idx is None: raise SystemExit("[err] No overlapping monitor for the table.")
        mon=sct.monitors[idx]; safe=clamp_to_monitor(rect,mon)
        if not safe: raise SystemExit("[err] Table outside selected monitor after clamping.")
        Ls,Ts,Rs,Bs=safe; W,H=Rs-Ls, Bs-Ts
        print(f"[info] Capturing {Ls,Ts,Rs,Bs} size={W}x{H} on monitor {idx}")

        px,py,pw,ph=place_preview_outside((Ls,Ts,Rs,Bs),mon)
        cv2.namedWindow("Card Recognition Test",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Card Recognition Test",pw,ph); cv2.moveWindow("Card Recognition Test",px,py)

        print("\nHotkeys:  S=save crops  G=toggle color-gate  Q/ESC=quit\n")
        last_report=""
        while True:
            raw=sct.grab({"left":Ls,"top":Ts,"width":W,"height":H})
            frame=np.array(raw)[:,:,:3].copy()

            guesses,scores={},{}
            for key in CARD_ROI_KEYS:
                box=clamp_box_to_wh(ROIS[key],W,H); x,y,w,h=box; roi=crop(frame,box)

                face_box=detect_card_face_box(roi,key)
                if face_box is not None:
                    fx,fy,fw,fh=face_box
                    cv2.rectangle(frame,(x+fx,y+fy),(x+fx+fw,y+fy+fh),(255,255,0),2)  # cyan
                    face_bgr=roi[fy:fy+fh, fx:fx+fw]
                else:
                    # amber dashed fallback
                    fw2,fh2=int(w*0.90),int(h*0.90); fx2,fy2=x+(w-fw2)//2, y+(h-fh2)//2
                    for i in range(0,fw2,10):
                        cv2.line(frame,(fx2+i,fy2),(fx2+i+5,fy2),(0,191,255),2)
                        cv2.line(frame,(fx2+i,fy2+fh2),(fx2+i+5,fy2+fh2),(0,191,255),2)
                    for j in range(0,fh2,10):
                        cv2.line(frame,(fx2,fy2+j),(fx2,fy2+j+5),(0,191,255),2)
                        cv2.line(frame,(fx2+fw2,fy2+j),(fx2+fw2,fy2+j+5),(0,191,255),2)
                    face_bgr=None

                glyph_only = key in ("hole1","hole2")

                if not card_present(roi,key):
                    label,score=None,0.0
                else:
                    label,score=recognize_card(roi,rank_multi,rank_edge_multi,suit_tm,full_gray,
                                               slot_name=key,glyph_only=glyph_only,face_bgr=face_bgr)

                guesses[key]=label; scores[key]=score
                color=(0,255,0) if label else (0,165,255)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,f"{key}:{label or '--'} ({scores[key]:.2f})",(x,y-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

            flop_ok=all(guesses[k] for k in ("board1","board2","board3"))
            if not flop_ok:
                for k in ("board4","board5"): guesses[k],scores[k]=None,0.0

            hole=f"{guesses['hole1'] or '--'} {guesses['hole2'] or '--'}"
            board=[guesses[k] for k in CARD_ROI_KEYS[2:]]
            line=f"Hole: {hole}   Board: {' '.join(b or '--' for b in board)}"
            if line!=last_report: print(line); last_report=line

            cv2.imshow("Card Recognition Test",frame)
            k=cv2.waitKey(30)&0xFFFF
            if k in (27,ord('q')): break
            if k==ord('s'):
                ts=int(time.time()*1000); dbg=f"debug_{ts}"; os.makedirs(dbg,exist_ok=True)
                for key in CARD_ROI_KEYS:
                    b=clamp_box_to_wh(ROIS[key],W,H); r=crop(frame,b)
                    cv2.imwrite(os.path.join(dbg,f"{key}.png"),r)
                    if r.size:
                        H2,W2=r.shape[:2]; base=BASES.get(key,(0.10,0.42,0.10,0.45))
                        y1,y2=int(base[0]*H2),int(base[1]*H2); x1,x2=int(base[2]*W2),int(base[3]*W2)
                        g=r[y1:y2,x1:x2]
                        if g.size: g=cv2.resize(g,(80,80)); cv2.imwrite(os.path.join(dbg,f"{key}_glyph.png"),g)
                print(f"[ok] wrote {dbg}/")
            if k in (ord('g'),ord('G')):
                COLOR_GATING_ENABLED = not COLOR_GATING_ENABLED
                print(f"[info] color gating: {'ON' if COLOR_GATING_ENABLED else 'OFF'}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
