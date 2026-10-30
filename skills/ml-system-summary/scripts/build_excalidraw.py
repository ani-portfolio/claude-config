#!/usr/bin/env python3
"""Build an Excalidraw diagram of an ML system from system_summary.json.

Usage:
    python3 build_excalidraw.py <system_summary.json> <output.excalidraw>

The layout is fixed so every diagram looks the same. Only the text changes.
Box heights grow with their content, and every box in a row shares one height.
Uses only the Python standard library.
"""

import json
import random
import sys
import time

# ---------------------------------------------------------------- style

FONT_FAMILY = 2          # Excalidraw "Helvetica" (normal, not hand drawn)
TITLE_SIZE = 16
BODY_SIZE = 13
LINE_HEIGHT = 1.25
PAD = 12
TITLE_GAP = 6
MAX_BULLETS = 7

COLORS = {
    "data": "#cfdcf8",       # blue: data and artifacts
    "pipeline": "#c3ecd0",   # green: pipelines
    "infra": "#f7cfcf",      # pink: stores, registry, deployment, monitoring
    "output": "#fff4bf",     # yellow: predictions and decisions
    "note": "#ffffff",       # white: framing columns
    "missing": "#eeeeee",    # grey: not found in codebase
}
STROKE = "#1e1e1e"
MUTED = "#6b6b6b"

# Helvetica advance widths (AFM units per 1000 em) for deterministic text wrapping.
_W = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191,
    "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
    ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556, "@": 1015, "[": 278,
    "\\": 278, "]": 278, "^": 469, "_": 556, "`": 333, "{": 334, "|": 260, "}": 334,
    "~": 584, "•": 350, "×": 584, "→": 1000, "–": 556, "—": 1000,
    "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778, "H": 722,
    "I": 278, "J": 500, "K": 667, "L": 556, "M": 833, "N": 722, "O": 778, "P": 667,
    "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944, "X": 667,
    "Y": 667, "Z": 611,
    "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556, "h": 556,
    "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556, "o": 556, "p": 556,
    "q": 556, "r": 333, "s": 500, "t": 278, "u": 556, "v": 500, "w": 722, "x": 500,
    "y": 500, "z": 500,
}
for _d in "0123456789":
    _W[_d] = 556


def text_width(s, size):
    return sum(_W.get(ch, 600) for ch in s) * size / 1000.0 * 1.04


def wrap(text, size, max_w):
    """Greedy word wrap. Long words are split by characters."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        cand = w if not cur else cur + " " + w
        if text_width(cand, size) <= max_w:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        while text_width(w, size) > max_w:
            cut = len(w)
            while cut > 1 and text_width(w[:cut], size) > max_w:
                cut -= 1
            lines.append(w[:cut])
            w = w[cut:]
        cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def bullet_lines(bullets, size, max_w):
    """Wrap bullets with a hanging indent so wrapped lines align after the bullet."""
    out = []
    indent = "   "
    for b in bullets:
        first = True
        for ln in wrap(str(b).strip(), size, max_w - text_width("• ", size)):
            out.append(("• " if first else indent) + ln)
            first = False
    return out


# ---------------------------------------------------------------- elements

_rng = random.Random(7)
_counter = [0]


def _id(prefix):
    _counter[0] += 1
    return f"{prefix}_{_counter[0]}"


def _base(kind, x, y, w, h, **kw):
    el = {
        "id": kw.pop("id", None) or _id(kind),
        "type": kind,
        "x": round(x, 2), "y": round(y, 2),
        "width": round(w, 2), "height": round(h, 2),
        "angle": 0,
        "strokeColor": kw.pop("strokeColor", STROKE),
        "backgroundColor": kw.pop("backgroundColor", "transparent"),
        "fillStyle": "solid",
        "strokeWidth": kw.pop("strokeWidth", 1),
        "strokeStyle": kw.pop("strokeStyle", "solid"),
        "roughness": 0,
        "opacity": 100,
        "groupIds": kw.pop("groupIds", []),
        "frameId": None,
        "roundness": None,
        "strokeSharpness": "sharp",
        "seed": _rng.randint(1, 2**31 - 1),
        "version": 1,
        "versionNonce": _rng.randint(1, 2**31 - 1),
        "isDeleted": False,
        "boundElements": kw.pop("boundElements", None),
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False,
    }
    el.update(kw)
    return el


def text_el(x, y, lines, size, color=STROKE, align="left", groupIds=None):
    content = "\n".join(lines)
    h = len(lines) * size * LINE_HEIGHT
    w = max(text_width(l, size) for l in lines) if lines else 0
    return _base(
        "text", x, y, w, h,
        strokeColor=color, groupIds=groupIds or [],
        text=content, originalText=content,
        fontSize=size, fontFamily=FONT_FAMILY,
        textAlign=align, verticalAlign="top",
        baseline=round(h - size * 0.28),
        containerId=None, lineHeight=LINE_HEIGHT, autoResize=True,
    )


class Box:
    """A titled box: rectangle plus title text plus body text, grouped together."""

    def __init__(self, key, title, bullets, kind, width, status="confirmed", max_bullets=MAX_BULLETS):
        self.key = key
        self.kind = kind
        self.status = status
        self.width = width
        suffix = {"inferred": " (inferred)", "documented": " (docs only, not in code)"}.get(status, "")
        self.title = title + suffix
        if status == "missing":
            bullets = ["Not found in codebase"]
        bullets = list(bullets or [])
        if max_bullets and len(bullets) > max_bullets:
            bullets = bullets[:max_bullets - 1] + [f"{len(bullets) - max_bullets + 1} more items in the summary"]
        inner = width - 2 * PAD
        self.title_lines = wrap(self.title, TITLE_SIZE, inner)
        self.body_lines = bullet_lines(bullets, BODY_SIZE, inner) if bullets else []
        self.content_h = (
            PAD
            + len(self.title_lines) * TITLE_SIZE * LINE_HEIGHT
            + (TITLE_GAP + len(self.body_lines) * BODY_SIZE * LINE_HEIGHT if self.body_lines else 0)
            + PAD
        )
        self.height = self.content_h
        self.x = self.y = 0
        self.rect_id = _id("box_" + key)

    # anchor points
    def left(self):
        return (self.x, self.y + self.height / 2)

    def right(self):
        return (self.x + self.width, self.y + self.height / 2)

    def top(self, x=None):
        return (self.x + self.width / 2 if x is None else x, self.y)

    def bottom(self, x=None):
        return (self.x + self.width / 2 if x is None else x, self.y + self.height)

    def elements(self):
        gid = _id("grp")
        missing = self.status in ("missing", "documented")
        rect = _base(
            "rectangle", self.x, self.y, self.width, self.height, id=self.rect_id,
            backgroundColor=COLORS["missing" if missing else self.kind],
            strokeColor=MUTED if missing else STROKE,
            strokeStyle="dashed" if missing else "solid",
            groupIds=[gid], boundElements=[],
        )
        els = [rect]
        color = MUTED if missing else STROKE
        ty = self.y + PAD
        els.append(text_el(self.x + PAD, ty, self.title_lines, TITLE_SIZE, color, groupIds=[gid]))
        if self.body_lines:
            by = ty + len(self.title_lines) * TITLE_SIZE * LINE_HEIGHT + TITLE_GAP
            els.append(text_el(self.x + PAD, by, self.body_lines, BODY_SIZE, color, groupIds=[gid]))
        return els


def arrow(points, start_box=None, end_box=None, dashed=False, boxes_by_rect=None):
    x0, y0 = points[0]
    rel = [[round(px - x0, 2), round(py - y0, 2)] for px, py in points]
    xs = [p[0] for p in rel]
    ys = [p[1] for p in rel]
    aid = _id("arrow")
    faded = any(b is not None and b.status in ("missing", "documented") for b in (start_box, end_box))
    el = _base(
        "arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys), id=aid,
        strokeStyle="dashed" if (dashed or faded) else "solid", strokeWidth=1.5,
        strokeColor=MUTED if faded else STROKE,
        points=rel, lastCommittedPoint=None,
        startBinding={"elementId": start_box.rect_id, "focus": 0, "gap": 1} if start_box else None,
        endBinding={"elementId": end_box.rect_id, "focus": 0, "gap": 1} if end_box else None,
        startArrowhead=None, endArrowhead="arrow", elbowed=False,
    )
    for b in (start_box, end_box):
        if b is not None:
            b.bound_arrows.append(aid)
    return el


# ---------------------------------------------------------------- layout

def comp(data, key):
    c = (data.get("components") or {}).get(key) or {}
    status = c.get("status", "missing")
    if status not in ("confirmed", "inferred", "documented", "missing"):
        status = "inferred"
    return c.get("bullets", []), status


def build(data):
    titles = {
        "raw_data": "Raw Data",
        "feature_pipeline": "Feature Pipeline",
        "engineered_features": "Engineered Features",
        "feature_store": "Feature Store",
        "training_pipeline": "Training Pipeline",
        "trained_model": "Trained Model",
        "model_registry": "Model Registry",
        "deployment_experimentation": "Deployment and Experimentation",
        "inference_pipeline": "Inference Pipeline",
        "predictions_decision": "Predictions and Decision",
        "monitoring": "Monitoring and Observability",
    }
    kinds = {
        "raw_data": "data", "engineered_features": "data", "trained_model": "data",
        "feature_pipeline": "pipeline", "training_pipeline": "pipeline", "inference_pipeline": "pipeline",
        "feature_store": "infra", "model_registry": "infra",
        "deployment_experimentation": "infra", "monitoring": "infra",
        "predictions_decision": "output",
    }
    W = 200
    widths = {
        "raw_data": W, "feature_pipeline": W, "engineered_features": 320,
        "feature_store": 520, "training_pipeline": W, "trained_model": W,
        "model_registry": W, "deployment_experimentation": 380,
        "inference_pipeline": 230, "predictions_decision": 220, "monitoring": 1420,
    }
    B = {}
    for k in titles:
        bullets, status = comp(data, k)
        B[k] = Box(k, titles[k], bullets, kinds[k], widths[k], status)
        B[k].bound_arrows = []

    # columns (x positions inside the system panel)
    X_RAW, X_FP, X_TP, X_IP = 0, 290, 590, 880
    X_PD = X_IP + 230 + 90
    B["raw_data"].x = X_RAW
    B["feature_pipeline"].x = X_FP
    B["engineered_features"].x = X_FP + W - 320
    B["feature_store"].x = X_TP
    B["training_pipeline"].x = X_TP
    B["trained_model"].x = X_TP
    B["model_registry"].x = X_TP
    B["deployment_experimentation"].x = X_TP + W + 80
    B["inference_pipeline"].x = X_IP
    B["predictions_decision"].x = X_PD
    B["monitoring"].x = X_RAW

    rows = [
        ["engineered_features", "feature_store"],
        ["raw_data", "feature_pipeline", "training_pipeline", "inference_pipeline", "predictions_decision"],
        ["trained_model"],
        ["model_registry", "deployment_experimentation"],
        ["monitoring"],
    ]
    gaps = [80, 70, 70, 90]
    y = 0
    for i, row in enumerate(rows):
        h = max(B[k].content_h for k in row)
        if row == ["trained_model"]:
            h = max(h, 60)
        for k in row:
            B[k].y = y
            B[k].height = h
        y += h + (gaps[i] if i < len(gaps) else 0)

    els = []
    arrows = []
    a = lambda pts, s, e, dashed=False: arrows.append(arrow(pts, s, e, dashed))

    raw, fp, ef, fs = B["raw_data"], B["feature_pipeline"], B["engineered_features"], B["feature_store"]
    tp, tm, mr, de = B["training_pipeline"], B["trained_model"], B["model_registry"], B["deployment_experimentation"]
    ip, pd, mon = B["inference_pipeline"], B["predictions_decision"], B["monitoring"]

    a([raw.right(), (fp.x, raw.right()[1])], raw, fp)
    fx = fp.x + fp.width / 2
    a([fp.top(fx), ef.bottom(fx)], fp, ef)
    a([ef.right(), (fs.x, ef.right()[1])], ef, fs)
    tx = tp.x + tp.width / 2
    ix = ip.x + ip.width / 2
    a([fs.bottom(tx), tp.top(tx)], fs, tp)
    a([fs.bottom(ix), ip.top(ix)], fs, ip)
    a([tp.bottom(tx), tm.top(tx)], tp, tm)
    a([tm.bottom(tx), mr.top(tx)], tm, mr)
    a([mr.right(), (de.x, mr.right()[1])], mr, de)
    a([de.top(ix), ip.bottom(ix)], de, ip)
    a([ip.right(), (pd.x, ip.right()[1])], ip, pd)

    loops = data.get("feedback_loops") or {}
    retrain = loops.get("retraining_trigger") or {}
    if retrain.get("present"):
        lx = tp.x - 50
        ty = tm.y + tm.height / 2
        a([(lx, mon.y), (lx, ty), (tm.x, ty)], mon, tm, dashed=True)
        label = wrap(retrain.get("label") or "Retraining trigger", BODY_SIZE, 230)
        lw = max(text_width(l, BODY_SIZE) for l in label)
        mid = (mon.y + ty) / 2 - len(label) * BODY_SIZE * LINE_HEIGHT / 2
        els.append(text_el(lx - lw - 10, mid, label, BODY_SIZE, MUTED, align="right"))

    behaviour = loops.get("behaviour_feedback") or {}
    loop_y = mon.y + mon.height + 50
    if behaviour.get("present"):
        rx = max(pd.x + pd.width, mon.x + mon.width) + 40
        pts = [pd.right(), (rx, pd.right()[1]), (rx, loop_y), (raw.x - 40, loop_y),
               (raw.x - 40, raw.y + raw.height / 2), raw.left()]
        a(pts, pd, raw, dashed=True)
        label = wrap(behaviour.get("label") or "Feedback loop", BODY_SIZE, 520)
        lw = max(text_width(l, BODY_SIZE) for l in label)
        els.append(text_el(mon.x + mon.width / 2 - lw / 2, loop_y + 8, label, BODY_SIZE, MUTED, align="center"))

    for b in B.values():
        bel = b.elements()
        bel[0]["boundElements"] = [{"id": i, "type": "arrow"} for i in b.bound_arrows]
        els = bel + els
    els += arrows

    # system panel frame
    xs, ys = [], []
    for e in els:
        if e["type"] == "arrow":
            xs += [e["x"] + p[0] for p in e["points"]]
            ys += [e["y"] + p[1] for p in e["points"]]
        else:
            xs += [e["x"], e["x"] + e["width"]]
            ys += [e["y"], e["y"] + e["height"]]
    sx0, sy0, sx1, sy1 = min(xs) - 50, min(ys) - 50, max(xs) + 50, max(ys) + 40
    frame_sys = _base("rectangle", sx0, sy0, sx1 - sx0, sy1 - sy0)

    # framing panel on the left (three note columns)
    fr = data.get("framing") or {}
    cols = [
        ("Business Goal", fr.get("business_goal", [])),
        ("Assumptions and Notes", fr.get("assumptions", [])),
        ("Problem Framing and Success Metrics", fr.get("problem_framing", [])),
    ]
    CW, CG = 250, 18
    nboxes = []
    for i, (t, bl) in enumerate(cols):
        nb = Box(f"framing{i}", t, bl or ["Not stated"], "note", CW, max_bullets=None)
        nb.bound_arrows = []
        nboxes.append(nb)
    ch = max(sy1 - sy0 - 80, max(n.content_h for n in nboxes))
    left_w = 3 * CW + 2 * CG
    fx0 = sx0 - 70 - left_w - 40
    for i, nb in enumerate(nboxes):
        nb.x = fx0 + 20 + i * (CW + CG)
        nb.y = sy0 + 40
        nb.height = ch
        els = nb.elements() + els
    frame_left = _base("rectangle", fx0, sy0, left_w + 40, ch + 80)

    # title
    title = data.get("system_name") or "ML System"
    sub = " · ".join(s for s in [data.get("repo"), data.get("generated_on")] if s)
    head = [text_el(fx0, sy0 - 64, [title], 24)]
    if sub:
        head.append(text_el(fx0, sy0 - 30, [sub], BODY_SIZE, MUTED))
    n = data.get("discrepancy_count")
    if isinstance(n, int) and n > 0:
        note = (f"{n} high impact discrepanc{'y' if n == 1 else 'ies'} between documentation and code. "
                "This diagram follows the code. See the summary for details.")
        head.append(text_el(sx0, sy0 - 30, [note], BODY_SIZE, "#b42318"))

    return [frame_left, frame_sys] + head + els


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)
    elements = build(data)
    scene = {
        "type": "excalidraw",
        "version": 2,
        "source": "ml-system-summary skill",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
        "files": {},
    }
    with open(sys.argv[2], "w") as f:
        json.dump(scene, f, indent=1, ensure_ascii=False)
    print(f"Wrote {sys.argv[2]} with {len(elements)} elements")


if __name__ == "__main__":
    main()
