#!/usr/bin/env python3
"""
KNICKS MASCOT — 30-second social media video
MoviePy 2.x + Pillow + numpy → MP4

Segments:
  HOOK    0–3 s   black screen, thunder flash, hook text, creature reveal
  BUILD   3–15 s  Ken Burns zoom, beat light leaks, 3 text overlays, glitch
  CLIMAX  15–25 s hard beat strobes, fast-cut labels, "KNICKS IN 6. NO CAP."
  CTA     25–30 s freeze, follow/tag calls-to-action
"""

import math
import os

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFont

# ─── Config ──────────────────────────────────────────────────────────────────
W, H   = 540, 960   # 9:16 portrait (TikTok / Reels)
FPS    = 24
DUR    = 30.0

ORANGE = (245, 132,  38)
BLUE   = (  0, 107, 182)
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)

DESKTOP = os.path.expanduser("~/Desktop")
os.makedirs(DESKTOP, exist_ok=True)
OUTPUT  = os.path.join(DESKTOP, "knicks_mascot.mp4")


# ─── Easing ──────────────────────────────────────────────────────────────────
def smoothstep(edge0, edge1, x):
    if edge1 == edge0:
        return float(x >= edge1)
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)

def ease_out(t):
    return 1.0 - (1.0 - t) ** 2


# ─── Fonts ───────────────────────────────────────────────────────────────────
def _load(size, bold=True):
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

FONT_HOOK   = _load(int(H * 0.038))
FONT_OVER   = _load(int(H * 0.048))
FONT_BIG    = _load(int(H * 0.072))
FONT_JERSEY = _load(int(H * 0.130))
FONT_CTA    = _load(int(H * 0.034))


# ─── Pre-computed grids (reused every frame) ──────────────────────────────────
_yy, _xx = np.mgrid[0:H, 0:W]                         # shape (H,W)
_cx_f, _cy_f = W / 2.0, H / 2.0
_dist_center = np.sqrt(((_xx - _cx_f) / (_cx_f))**2   # normalised 0..√2
                      + ((_yy - _cy_f) / (_cy_f))**2)
_dist_left   = np.sqrt(((_xx       ) / W)**2 + ((_yy - H/2) / H)**2)
_dist_right  = np.sqrt(((_xx - W   ) / W)**2 + ((_yy - H/2) / H)**2)
_VIG         = np.clip(1.0 - _dist_center * 0.55, 0.12, 1.0)[..., np.newaxis]


# ─── Creature (pre-rendered RGBA PIL image) ───────────────────────────────────
def _build_creature() -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    cx  = W // 2

    # Outer aura glow
    for i in range(28, 0, -2):
        a = int(3.5 * (29 - i))
        d.ellipse(
            [int(W*0.12) - i*3, int(H*0.14) - i*2,
             int(W*0.88) + i*3, int(H*0.86) + i*2],
            fill=(*ORANGE, a),
        )

    # Body silhouette
    d.ellipse(
        [int(W*0.14), int(H*0.15), int(W*0.86), int(H*0.85)],
        fill=(16, 5, 0, 248),
    )

    # ── Eyes ──────────────────────────────────────────
    EY1, EY2 = int(H*0.27), int(H*0.38)
    EW       = int(W*0.14)
    EY_mid   = (EY1 + EY2) // 2
    for ex in (int(W*0.33), int(W*0.67)):
        # halo rings
        for i in range(14, 0, -2):
            d.ellipse(
                [ex - EW//2 - i, EY1 + EW//4 - i,
                 ex + EW//2 + i, EY2 - EW//4 + i],
                fill=(*ORANGE, int(10 * (15 - i))),
            )
        # iris
        d.ellipse(
            [ex - EW//2, EY1 + EW//4, ex + EW//2, EY2 - EW//4],
            fill=(*ORANGE, 255),
        )
        # pupil
        PR = int(EW * 0.30)
        d.ellipse([ex-PR, EY_mid-PR, ex+PR, EY_mid+PR], fill=(0, 0, 0, 255))
        # shine
        SR = max(2, PR//3)
        d.ellipse(
            [ex - PR + SR, EY_mid - PR + SR//2,
             ex - PR + SR*3, EY_mid - PR + SR*2],
            fill=(255, 255, 255, 200),
        )

    # ── Mouth & fangs ─────────────────────────────────
    my = int(H*0.44)
    d.arc([cx-46, my-12, cx+46, my+34], start=8, end=172,
          fill=(*ORANGE, 155), width=3)
    for fx, sign in ((cx-28, -1), (cx+8, 1)):
        d.polygon(
            [(fx, my), (fx+10, my+26), (fx+20, my)],
            fill=(225, 225, 225, 218),
        )

    # ── Jersey #33 ────────────────────────────────────
    ny = int(H*0.51)
    # shadow
    bbox = d.textbbox((0, 0), "33", font=FONT_JERSEY)
    nw   = bbox[2] - bbox[0]
    nx   = cx - nw // 2
    d.text((nx+3, ny+3), "33", font=FONT_JERSEY, fill=(170, 55, 0, 110))
    d.text((nx,   ny),   "33", font=FONT_JERSEY, fill=(*ORANGE, 212))

    # ── Right arm (pointing) ───────────────────────────
    d.polygon(
        [(int(W*0.79), int(H*0.45)),
         (int(W*0.97), int(H*0.49)),
         (int(W*0.96), int(H*0.54)),
         (int(W*0.78), int(H*0.51))],
        fill=(16, 5, 0, 238),
    )
    # claws
    for angle_deg in (-28, -6, 18):
        ang = math.radians(angle_deg)
        px, py = int(W*0.97), int(H*0.50)
        d.line(
            [(px, py), (px + int(math.cos(ang)*22), py - int(math.sin(ang)*22))],
            fill=(*ORANGE, 205), width=2,
        )

    # ── Left arm (hanging) ────────────────────────────
    d.polygon(
        [(int(W*0.21), int(H*0.47)),
         (int(W*0.03), int(H*0.60)),
         (int(W*0.07), int(H*0.65)),
         (int(W*0.23), int(H*0.53))],
        fill=(16, 5, 0, 238),
    )

    # ── Legs ──────────────────────────────────────────
    for pts in (
        [(int(W*0.32), int(H*0.83)), (int(W*0.24), int(H*0.96)),
         (int(W*0.38), int(H*0.96)), (int(W*0.43), int(H*0.83))],
        [(int(W*0.57), int(H*0.83)), (int(W*0.62), int(H*0.96)),
         (int(W*0.76), int(H*0.96)), (int(W*0.67), int(H*0.83))],
    ):
        d.polygon(pts, fill=(16, 5, 0, 238))

    return img


print("Pre-rendering creature…")
_CREATURE_PIL = _build_creature()

# Pre-bake zoom cache  (1.0 → 1.40 sampled at every frame's rounded zoom)
print("Pre-baking zoom cache…")
_ZOOM_CACHE: dict[int, np.ndarray] = {}

for _fi in range(int(DUR * FPS) + 1):
    _t   = _fi / FPS
    _z   = 1.0 + smoothstep(0, DUR, _t) * 0.40
    _key = round(_z * 1000)                    # integer key, e.g. 1000..1400
    if _key not in _ZOOM_CACHE:
        _nw = int(W * _z)
        _nh = int(H * _z)
        _sc = _CREATURE_PIL.resize((_nw, _nh), Image.BILINEAR)
        _x0 = (_nw - W) // 2
        _y0 = (_nh - H) // 2
        _ZOOM_CACHE[_key] = np.array(_sc.crop((_x0, _y0, _x0+W, _y0+H)))

print(f"  cached {len(_ZOOM_CACHE)} zoom levels")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _zoom_arr(t: float) -> np.ndarray:
    """Return RGBA numpy array of the zoomed creature."""
    z   = 1.0 + smoothstep(0, DUR, t) * 0.40
    key = round(z * 1000)
    # nearest cached key
    if key not in _ZOOM_CACHE:
        key = min(_ZOOM_CACHE.keys(), key=lambda k: abs(k - key))
    return _ZOOM_CACHE[key]                                   # shape (H,W,4)


def _composite(base: np.ndarray, overlay_rgba: np.ndarray,
               alpha_mult: float = 1.0) -> np.ndarray:
    """Alpha-composite RGBA overlay onto RGB base, both uint8."""
    a = overlay_rgba[..., 3:4].astype(np.float32) / 255.0 * alpha_mult
    r = base.astype(np.float32) * (1 - a) + overlay_rgba[..., :3].astype(np.float32) * a
    return np.clip(r, 0, 255).astype(np.uint8)


def _add_light(base: np.ndarray, dist_map: np.ndarray,
               color: tuple, radius: float, strength: float) -> np.ndarray:
    mask = np.clip(1.0 - dist_map / radius, 0.0, 1.0) ** 2 * strength
    r = base.astype(np.float32) + np.stack(
        [mask * color[0], mask * color[1], mask * color[2]], axis=-1
    )
    return np.clip(r, 0, 255).astype(np.uint8)


def _scanlines(frame: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32)
    out[::2] *= 0.88
    return out.clip(0, 255).astype(np.uint8)


def _vignette(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * _VIG, 0, 255).astype(np.uint8)


def _glitch(frame: np.ndarray, intensity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed % (2**31))
    out = frame.copy()
    for _ in range(int(2 + intensity * 5)):
        y     = int(rng.integers(0, H - 20))
        hbar  = int(rng.integers(2, max(3, int(14 * intensity))))
        shift = int(rng.integers(-22, 22) * intensity)
        out[y:y+hbar, :, 0] = np.roll(frame[y:y+hbar, :, 0],  shift,      axis=1)
        out[y:y+hbar, :, 1] = np.roll(frame[y:y+hbar, :, 1], -shift // 2, axis=1)
        out[y:y+hbar, :, 2] = np.roll(frame[y:y+hbar, :, 2],  shift // 3, axis=1)
    return out


def _text_size(d: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = d.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_centered(d: ImageDraw.ImageDraw, y: int, text: str, font,
                   color: tuple, alpha: int = 255) -> int:
    tw, th = _text_size(d, text, font)
    tx = W // 2 - tw // 2
    d.text((tx+2, y+2), text, font=font, fill=(0, 0, 0, min(255, alpha // 2)))
    d.text((tx,   y),   text, font=font, fill=(*color, alpha))
    return th


def _draw_sidebar(d: ImageDraw.ImageDraw, y: int, text: str, font,
                  bar_color: tuple, text_color: tuple,
                  alpha: int = 255, x_offset: int = 0) -> int:
    tw, th = _text_size(d, text, font)
    px     = 22 + x_offset
    pad    = 10
    d.rectangle([px,    y-pad,  px+4,         y+th+pad], fill=(*bar_color,  alpha))
    d.rectangle([px+6,  y-pad,  px+6+tw+pad*2, y+th+pad],
                fill=(0, 0, 0, min(255, alpha * 7 // 10)))
    d.text((px+6+pad, y), text, font=font, fill=(*text_color, alpha))
    return th + pad * 2


# ─── Frame generator ─────────────────────────────────────────────────────────
def make_frame(t: float) -> np.ndarray:

    # ══════════════════════════════════════════════════
    #  HOOK  0 – 3 s
    # ══════════════════════════════════════════════════
    if t < 3.0:
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Thunder flash at ~0.7 s
        if 0.60 < t < 0.90:
            fv = int(255 * (1.0 - abs(t - 0.75) / 0.15))
            frame[:] = fv

        # Background + creature fade in 1.8 → 3.0
        reveal = smoothstep(1.8, 3.0, t)
        if reveal > 0.0:
            bg_ov   = np.zeros((H, W, 3), dtype=np.uint8)
            bg_ov   = _add_light(bg_ov, _dist_left,  ORANGE, 0.9, 0.12 * reveal)
            bg_ov   = _add_light(bg_ov, _dist_right, BLUE,   0.9, 0.08 * reveal)
            frame   = np.clip(
                frame.astype(np.float32) * (1-reveal) + bg_ov.astype(np.float32) * reveal,
                0, 255,
            ).astype(np.uint8)
            frame   = _composite(frame, _zoom_arr(t), alpha_mult=reveal)

        # Text overlay
        img = Image.fromarray(frame).convert("RGBA")
        d   = ImageDraw.Draw(img)

        txt_a = int(255 * smoothstep(0.85, 1.35, t) * (1.0 - smoothstep(2.4, 3.0, t)))
        if txt_a > 5:
            for i, line in enumerate(['"They said the Knicks', 'needed a new mascot..."']):
                y = int(H * 0.36) + i * (int(H * 0.048) + 6)
                tw, _ = _text_size(d, line, FONT_HOOK)
                tx = W // 2 - tw // 2
                d.text((tx+2, y+2), line, font=FONT_HOOK, fill=(0, 0, 0, txt_a // 2))
                d.text((tx,   y),   line, font=FONT_HOOK, fill=(255, 255, 255, txt_a))

        frame = np.array(img.convert("RGB"))
        return _vignette(frame)

    # ══════════════════════════════════════════════════
    #  BUILD  3 – 15 s
    # ══════════════════════════════════════════════════
    elif t < 15.0:
        bt   = t - 3.0                                  # 0 → 12
        beat = (math.sin(bt * 1.6 * 2 * math.pi) + 1) / 2   # ~96 BPM pulse

        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg = _add_light(bg, _dist_left,  ORANGE, 0.85, 0.14 + beat * 0.18)
        bg = _add_light(bg, _dist_right, BLUE,   0.85, 0.09 + (1-beat) * 0.14)
        bg = _composite(bg, _zoom_arr(t))
        bg = _scanlines(bg)

        # Glitch every ~3 s
        gp = (math.sin(bt * 0.7 * math.pi) + 1) / 2
        if gp > 0.86:
            bg = _glitch(bg, 0.35 + gp * 0.3, seed=int(t * 40))

        img = Image.fromarray(bg).convert("RGBA")
        d   = ImageDraw.Draw(img)

        overlays = [
            # (start, end, lines,                         bar_col, txt_col)
            (1.0,  13.0, ['"Born in the sewers',
                           'of Manhattan..."'],           ORANGE, ORANGE),
            (4.5,  13.0, ['"Raised on MSG hardwood..."'], BLUE,   (100, 180, 255)),
            (8.0,  13.0, ['"He bleeds orange & blue..."'],ORANGE, ORANGE),
        ]
        BASE_Y = [int(H*0.07), int(H*0.22), int(H*0.37)]

        for idx, (start, end, lines, bar_col, txt_col) in enumerate(overlays):
            if bt < start:
                continue
            fade_in  = smoothstep(start, start + 0.35, bt)
            fade_out = 1.0 - smoothstep(end, end + 0.25, bt)
            alpha    = int(255 * fade_in * fade_out)
            if alpha < 5:
                continue
            x_slide  = int((1.0 - ease_out(smoothstep(start, start+0.4, bt))) * 38)
            y_pos    = BASE_Y[idx]
            for line in lines:
                rh = _draw_sidebar(d, y_pos, line, FONT_OVER,
                                   bar_col, txt_col, alpha=alpha, x_offset=x_slide)
                y_pos += rh + 4

        frame = np.array(img.convert("RGB"))
        return _vignette(frame)

    # ══════════════════════════════════════════════════
    #  CLIMAX  15 – 25 s
    # ══════════════════════════════════════════════════
    elif t < 25.0:
        ct        = t - 15.0                                       # 0 → 10
        fast_beat = (math.sin(ct * 2.5 * 2 * math.pi) + 1) / 2   # ~150 BPM
        hard      = fast_beat ** 3

        # Colour-flashing background
        if hard > 0.65:
            bg = np.clip(
                np.full((H, W, 3), [int(28*hard), int(7*hard), 0], np.float32),
                0, 255,
            ).astype(np.uint8)
            bg = _add_light(bg, _dist_left, ORANGE, 0.65, hard * 0.55)
        else:
            bg = np.clip(
                np.full((H, W, 3), [0, int(8*(1-hard)), int(22*(1-hard))], np.float32),
                0, 255,
            ).astype(np.uint8)
            bg = _add_light(bg, _dist_right, BLUE, 0.65, (1-fast_beat) * 0.45)

        bg = _composite(bg, _zoom_arr(t))

        # Heavy glitch on the beat
        if hard > 0.25:
            bg = _glitch(bg, 0.45 + hard * 1.6, seed=int(t * 120))

        # White flash on peak beats
        if hard > 0.90:
            fv = int((hard - 0.90) / 0.10 * 160)
            bg = np.clip(bg.astype(np.int16) + fv, 0, 255).astype(np.uint8)

        bg = _scanlines(bg)

        img = Image.fromarray(bg).convert("RGBA")
        d   = ImageDraw.Draw(img)

        # Fast-cut labels (one per 2.5 s window)
        cuts = [
            (0.0,  2.5, "CLOSE-UP EYES"),
            (2.5,  5.0, "JERSEY  #33"),
            (5.0,  7.5, "THE POINT"),
            (7.5, 10.0, "MSG"),
        ]
        for cs, ce, label in cuts:
            if cs <= ct < ce:
                lt    = ct - cs
                dur_c = ce - cs
                a_in  = smoothstep(0.0, 0.12, lt)
                a_out = 1.0 - smoothstep(dur_c - 0.18, dur_c, lt)
                alpha = int(255 * a_in * a_out)
                if alpha > 5:
                    _draw_centered(d, int(H*0.41), label, FONT_BIG, ORANGE, alpha)
                break

        # "KNICKS IN 6. NO CAP." fades in at ct > 6.5
        if ct > 6.5:
            a = int(255 * smoothstep(6.5, 8.0, ct))
            if a > 5:
                _draw_centered(d, int(H*0.70), "KNICKS IN 6.",  FONT_BIG, ORANGE,  a)
                _draw_centered(d, int(H*0.70) + int(H*0.08)+8,
                               "NO CAP.",                        FONT_BIG, WHITE,   a)

        frame = np.array(img.convert("RGB"))
        return _vignette(frame)

    # ══════════════════════════════════════════════════
    #  CTA  25 – 30 s
    # ══════════════════════════════════════════════════
    else:
        ft = t - 25.0   # 0 → 5

        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg = _add_light(bg, _dist_left,  ORANGE, 0.9, 0.18)
        bg = _add_light(bg, _dist_right, BLUE,   0.9, 0.12)
        bg = _composite(bg, _zoom_arr(t))
        # Darkened "freeze" feel
        bg = np.clip(bg.astype(np.float32) * 0.70, 0, 255).astype(np.uint8)
        bg = _scanlines(bg)

        img = Image.fromarray(bg).convert("RGBA")
        d   = ImageDraw.Draw(img)

        # Keep climax text
        _draw_centered(d, int(H*0.70), "KNICKS IN 6.", FONT_BIG, ORANGE, 200)
        _draw_centered(d, int(H*0.70) + int(H*0.08)+8, "NO CAP.", FONT_BIG, WHITE, 200)

        # CTA 1
        a1 = int(255 * smoothstep(0.4, 1.1, ft))
        if a1 > 5:
            for i, line in enumerate([
                "Follow if you want him",
                "as the official mascot",
            ]):
                tw, th = _text_size(d, line, FONT_CTA)
                y      = int(H*0.84) + i * (th + 8)
                d.rectangle(
                    [W//2 - tw//2 - 14, y-6, W//2 - tw//2 - 10, y+th+6],
                    fill=(*ORANGE, a1),
                )
                d.rectangle(
                    [W//2 - tw//2 - 10, y-6, W//2 + tw//2 + 12, y+th+6],
                    fill=(0, 0, 0, min(255, a1 * 8 // 10)),
                )
                d.text((W//2 - tw//2, y), line, font=FONT_CTA,
                       fill=(255, 255, 255, a1))

        # CTA 2
        a2 = int(255 * smoothstep(1.1, 1.9, ft))
        if a2 > 5:
            line = "Tag a Knicks fan who needs this"
            tw, th = _text_size(d, line, FONT_CTA)
            y      = int(H*0.92)
            d.rectangle(
                [W//2 - tw//2 - 14, y-6, W//2 - tw//2 - 10, y+th+6],
                fill=(*BLUE, a2),
            )
            d.rectangle(
                [W//2 - tw//2 - 10, y-6, W//2 + tw//2 + 12, y+th+6],
                fill=(0, 0, 0, min(255, a2 * 8 // 10)),
            )
            d.text((W//2 - tw//2, y), line, font=FONT_CTA,
                   fill=(255, 255, 255, a2))

        frame = np.array(img.convert("RGB"))
        return _vignette(frame)


# ─── Render ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nRendering {int(DUR)}s × {FPS}fps = {int(DUR*FPS)} frames  →  {OUTPUT}\n")
    clip = VideoClip(make_frame, duration=DUR)
    clip.write_videofile(
        OUTPUT,
        fps=FPS,
        codec="libx264",
        preset="fast",
        threads=4,
        audio=False,
        logger="bar",
    )
    print(f"\nDone!  Saved to: {OUTPUT}")
