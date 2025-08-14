# poker_bot.py
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Literal, Optional

import eval7
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ----------------------------
# Constants & simple utilities
# ----------------------------
ActionStr = Literal["fold", "call", "raise", "check"]
RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades

def _card(s: str) -> eval7.Card:
    return eval7.Card(s)

def _hand(s: str) -> Tuple[eval7.Card, eval7.Card]:
    assert len(s) == 4, f"hand must be 4 chars like 'AhKd', got {s}"
    return _card(s[:2]), _card(s[2:])

def _dead_set(hero: Tuple[eval7.Card, eval7.Card], board: List[str]) -> set:
    dead = {hero[0], hero[1]}
    for b in board:
        if b:
            dead.add(_card(b))
    return dead

# ----------------------------
# Range parsing helpers
# ----------------------------
def expand_dash_tokens(tokens: List[str]) -> List[str]:
    """
    Expands tokens like '22-77' -> ['22','33','44','55','66','77']
    and 'A2s-A5s' -> ['A2s','A3s','A4s','A5s'].
    Leaves normal tokens unchanged.
    """
    out = []
    for t in tokens:
        if "-" not in t:
            out.append(t); continue
        a, b = t.split("-", 1)
        # Pair run? e.g., 22-77
        if len(a) == 2 and a[0] == a[1] and len(b) == 2 and b[0] == b[1]:
            start = RANKS.index(a[0]); end = RANKS.index(b[0])
            step = 1 if end >= start else -1
            for i in range(start, end + step, step):
                r = RANKS[i]
                out.append(r + r)
            continue
        # Ax suited/offsuit run? e.g., A2s-A5s
        if len(a) == 3 and len(b) == 3 and a[0] == b[0] and a[2] == b[2]:
            hi = a[0]; suit = a[2]
            start = RANKS.index(a[1]); end = RANKS.index(b[1])
            step = 1 if end >= start else -1
            for i in range(start, end + step, step):
                out.append(hi + RANKS[i] + suit)
            continue
        out.append(t)
    return out

def expand_range(tokens: List[str]) -> List[Tuple[eval7.Card, eval7.Card]]:
    """
    Supports tokens like pairs '77', 'TT+',
    suited 'A2s+', 'KTs+', exact 'QJs',
    offsuit 'ATo+', 'KQo', and dash ranges '22-77', 'A2s-A5s'.
    """
    tokens = expand_dash_tokens(tokens)
    combos = set()

    def add_specific(r1, r2, suited_state: str):
        # suited_state: "s" (suited only), "o" (offsuit only), "both" (any)
        for s1 in SUITS:
            for s2 in SUITS:
                if r1 != r2:
                    if suited_state == "o" and s1 == s2:
                        continue
                    if suited_state == "s" and s1 != s2:
                        continue
                if r1 == r2 and s1 >= s2:
                    continue  # avoid duplicate pair permutations
                c1 = eval7.Card(r1 + s1)
                c2 = eval7.Card(r2 + s2)
                if c1 == c2:
                    continue
                # canonical ordering to reduce dupes
                if RANKS.index(r1) < RANKS.index(r2):
                    t = (c2, c1)
                else:
                    t = (c1, c2)
                combos.add(tuple(sorted(t, key=lambda c: (c.rank, c.suit))))

    def add_pair_range(start_rank: str):
        start_i = RANKS.index(start_rank)
        for i in range(start_i, len(RANKS)):
            r = RANKS[i]
            add_specific(r, r, suited_state="both")

    def add_ax_plus(high: str, suited_state: str):
        # e.g., A2s+ -> A2s..AKs, ATo+ -> ATo..AKo
        hi = RANKS.index(high)
        for i in range(0, hi + 1):
            r = RANKS[i]
            if r == "A":
                continue
            add_specific("A", r, suited_state)

    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if len(t) == 2 and t[0] == t[1]:  # "77"
            add_specific(t[0], t[1], suited_state="both")
        elif len(t) == 3 and t[0] == t[1] and t[2] == "+":  # "TT+"
            add_pair_range(t[0])
        elif len(t) in (3, 4):
            r1, r2 = t[0], t[1]
            tail = t[2:]
            if tail in ("s", "o"):
                add_specific(r1, r2, suited_state=tail)
            elif tail in ("s+", "o+"):
                suited_state = tail[0]
                start = r2
                start_i = RANKS.index(start)
                for j in range(start_i, len(RANKS)):
                    rr = RANKS[j]
                    if rr == r1:  # skip AA formed accidentally
                        continue
                    add_specific(r1, rr, suited_state=suited_state)
            elif tail == "+" and r1 == "A":
                # "A2+" ~ treat as both suited+offsuit climbs from the kicker
                add_ax_plus(r2, suited_state="s")
                add_ax_plus(r2, suited_state="o")
            else:
                # exact like "KQo" or "QJs" or ambiguous "KQ"
                suited_state = "o" if tail == "o" else "s" if tail == "s" else "both"
                if suited_state == "both":
                    add_specific(r1, r2, "s")
                    add_specific(r1, r2, "o")
                else:
                    add_specific(r1, r2, suited_state)
        else:
            # ignore unknown tokens
            pass

    return list(combos)

def filter_dead(combos, dead: set):
    return [(c1, c2) for (c1, c2) in combos if c1 not in dead and c2 not in dead]

# ----------------------------
# Equity via Monte Carlo
# ----------------------------
def monte_carlo_equity(hero: Tuple[eval7.Card, eval7.Card],
                       board_cards: List[str],
                       villain_range_tokens: List[str],
                       iters: int = 4000) -> float:
    deck = [eval7.Card(r + s) for r in RANKS for s in SUITS]
    dead = _dead_set(hero, board_cards)
    deck = [c for c in deck if c not in dead]

    board = [_card(b) for b in board_cards if b]
    vr_all = expand_range(villain_range_tokens)
    vr = filter_dead(vr_all, dead)
    if not vr:
        return 0.0

    score = 0.0
    need = 5 - len(board)

    for _ in range(iters):
        v1, v2 = random.choice(vr)
        used = {hero[0], hero[1], v1, v2, *board}
        rem = [c for c in deck if c not in {v1, v2}]
        drawn = []
        if need > 0:
            drawn = random.sample([c for c in rem if c not in used], need)
        full_board = board + drawn

        h_val = eval7.evaluate([hero[0], hero[1]] + full_board)
        v_val = eval7.evaluate([v1, v2] + full_board)

        if h_val > v_val:
            score += 1
        elif h_val == v_val:
            score += 0.5

    return score / iters

# ----------------------------
# Heuristic decision (MVP)
# ----------------------------
def choose_action(*,
                  hero_hand_str: str,
                  board_strs: List[str],
                  pot: float,
                  to_call: float,
                  effective_stack: float,
                  villain_range_tokens: List[str],
                  street: Literal["preflop", "flop", "turn", "river"]) -> Dict:
    h = _hand(hero_hand_str)
    equity = monte_carlo_equity(h, board_strs, villain_range_tokens, iters=4000)
    pot_odds = (to_call / (pot + to_call)) if to_call > 0 else 0.0

    if to_call == 0:
        action: ActionStr = "check"
        size = 0.0
        if street in ("flop", "turn") and equity >= 0.55 and effective_stack > 0.5 * pot:
            action = "raise"   # bet
            size = round(0.33 * pot, 2)
        return {"action": action, "size": size, "equity": equity, "pot_odds": pot_odds}

    if equity + 0.01 < pot_odds:
        return {"action": "fold", "size": 0.0, "equity": equity, "pot_odds": pot_odds}

    surplus = equity - pot_odds
    if surplus >= 0.12 and effective_stack > 2.0 * to_call:
        raise_to = round(pot + 3 * to_call, 2)  # simple heuristic
        return {"action": "raise", "size": raise_to, "equity": equity, "pot_odds": pot_odds}

    return {"action": "call", "size": to_call, "equity": equity, "pot_odds": pot_odds}

# ----------------------------
# CSV loader & mapper (ALL positions)
# ----------------------------
RANGES_DIR = Path(__file__).parent / "ranges" / "9max"

def load_tokens_from_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []
    tokens = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            tokens.append(t)
    return tokens

_POSITIONS = {"UTG","MP","HJ","CO","BTN","SB","BB"}

def _open_csv_for(pos: str) -> List[str]:
    path = RANGES_DIR / "preflop_open" / f"{pos}_open.csv"
    return load_tokens_from_csv(path)

def _defend_csv_for(defender_pos: str, opener_pos: str, kind: str) -> List[str]:
    """
    kind: 'call' or '3bet'
    Uses files: ranges/9max/defend_vs_open/<DEFENDER>_vs_<OPENER>_<kind>.csv
    Example: BTN_vs_HJ_call.csv
    """
    path = RANGES_DIR / "defend_vs_open" / f"{defender_pos}_vs_{opener_pos}_{kind}.csv"
    return load_tokens_from_csv(path)

def _postflop_generic() -> List[str]:
    return load_tokens_from_csv(RANGES_DIR / "postflop" / "srp_generic.csv")

Position = Literal["UTG","MP","HJ","CO","BTN","SB","BB"]
PreAction = Literal["open","call","3bet","4bet","raise","limp","fold"]

class Action(BaseModel):
    pos: Position
    act: PreAction
    size_bb: Optional[float] = None  # optional, not used yet

class RangeContext(BaseModel):
    street: Literal["preflop","flop","turn","river"]
    hero_pos: Position
    villain_pos: Position
    action_history: List[Action] = Field(default_factory=list)

def estimate_range_from_history(ctx: RangeContext) -> List[str]:
    """
    Chooses villain range tokens from action history.

    Implemented:
      • Preflop opener ranges for UTG/MP/HJ/CO/BTN/SB
      • Universal Defend-vs-Open for ANY defender in {MP,HJ,CO,BTN,SB,BB}
        against ANY opener in {UTG,MP,HJ,CO,BTN,SB} using <DEFENDER>_vs_<OPENER>_{call|3bet}.csv
      • Postflop: generic SRP fallback
    """
    # -------- Postflop --------
    if ctx.street in ("flop","turn","river"):
        toks = _postflop_generic()
        return toks if toks else ["22+","A2s+","K9s+","QTs+","JTs","ATo+","KQo"]

    # -------- Preflop --------
    # find the first open in the sequence
    opener = next((a for a in ctx.action_history if a.act == "open"), None)

    # A) Villain IS the opener -> use their open chart
    if opener and opener.pos == ctx.villain_pos and ctx.villain_pos in _POSITIONS:
        toks = _open_csv_for(ctx.villain_pos)
        if toks: return toks

    # B) Villain is some other position defending vs that opener
    if opener and (ctx.villain_pos in _POSITIONS) and (opener.pos in _POSITIONS):
        # Did villain 3bet already?
        villain_3bet = any(a.pos == ctx.villain_pos and a.act in ("3bet","raise") for a in ctx.action_history)
        kind = "3bet" if villain_3bet else "call"
        toks = _defend_csv_for(ctx.villain_pos, opener.pos, kind)
        if toks: return toks

    # C) Fallbacks: try villain open (if exists), then generic
    if ctx.villain_pos in _POSITIONS:
        toks = _open_csv_for(ctx.villain_pos)
        if toks: return toks

    return ["22+","A2s+","K9s+","QTs+","JTs","ATo+","KQo"]

# ----------------------------
# FastAPI models & endpoint
# ----------------------------
class AdviseRequest(BaseModel):
    hero_hand: str
    board: List[str] = []
    pot: float
    to_call: float
    effective_stack: float
    street: Literal["preflop", "flop", "turn", "river"]

    # Option A: explicit range
    villain_range: Optional[List[str]] = None

    # Option B: infer from action history
    hero_pos: Optional[Position] = None
    villain_pos: Optional[Position] = None
    action_history: Optional[List[Action]] = None

class AdviseResponse(BaseModel):
    action: ActionStr
    size: float
    equity: float
    pot_odds: float

app = FastAPI()

@app.post("/advise", response_model=AdviseResponse)
def advise(req: AdviseRequest):
    if req.villain_range:
        vr = req.villain_range
    elif req.villain_pos and req.hero_pos and req.action_history is not None:
        vr = estimate_range_from_history(
            RangeContext(
                street=req.street,
                hero_pos=req.hero_pos,
                villain_pos=req.villain_pos,
                action_history=[Action(**a) if isinstance(a, dict) else a
                                for a in req.action_history],
            )
        )
    else:
        vr = ["22+","A2s+","K9s+","QTs+","JTs","ATo+","KQo"]

    out = choose_action(
        hero_hand_str=req.hero_hand,
        board_strs=req.board,
        pot=req.pot,
        to_call=req.to_call,
        effective_stack=req.effective_stack,
        street=req.street,
        villain_range_tokens=vr,
    )
    return AdviseResponse(**out)

if __name__ == "__main__":
    # quick manual test
    example = advise(AdviseRequest(
        hero_hand="AhKd",
        board=["7h","8d","2c"],
        pot=10.0,
        to_call=3.0,
        effective_stack=100.0,
        street="flop",
        hero_pos="MP",
        villain_pos="UTG",
        action_history=[Action(pos="UTG", act="open", size_bb=2.5)]
    ))
    print(example.model_dump())
