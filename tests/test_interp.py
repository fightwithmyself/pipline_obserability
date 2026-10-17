"""运动补偿补帧与结构保真度指标。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from pipeline.interp import blend, estimate_flow, interpolate
from pipeline.steps.anime import structure_scores

W, H = 480, 288


def scene(x: int) -> np.ndarray:
    """平滑纹理背景(静止) + 一个带内部纹理的方块(模拟运动的人物),左上角在 (x, 90)。"""
    rng = np.random.default_rng(0)
    low = rng.integers(40, 200, (9, 15)).astype(np.uint8)
    bg = np.asarray(Image.fromarray(low).resize((W, H), Image.BICUBIC)).astype(np.int32)
    bg = (bg + rng.integers(-6, 7, (H, W))).clip(0, 255)
    im = np.stack([bg, bg, bg], axis=2).astype(np.uint8)
    yy, xx = np.mgrid[0:96, 0:96]
    patch = np.stack([200 + 50 * np.sin(xx / 7.0), 60 + 120 * (yy / 96.0), 40 + 100 * np.cos((xx + yy) / 9.0)], axis=2)
    im[90:186, x:x + 96] = np.clip(patch, 0, 255).astype(np.uint8)
    return im


def mse(a, b):
    return float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())


def test_flow_recovers_translation():
    a, b = scene(100), scene(140)                       # 方块向右平移 40px
    g = lambda im: im.astype(np.float32).mean(axis=2)
    dx, dy, s = estimate_flow(g(b), g(a))               # b 里的像素 x 对应 a 里的 x-40
    box = (slice(110 // s, 170 // s), slice(160 // s, 220 // s))      # 方块内部(1/s 分辨率)
    assert abs(float(np.median(dx[box])) * s + 40) < 6
    assert abs(float(np.median(dy[box])) * s) < 6


def test_motion_compensation_beats_blend_on_moving_object():
    a, b, f = scene(60), scene(180), scene(120)         # 方块在 a、b 之间平移 120px,f 是正中间
    mc = interpolate(a, b, a, b, f, 0.5)
    bl = blend(a, b, 0.5)
    assert mc.shape == a.shape and mc.dtype == np.uint8
    assert mse(mc, f) < 0.5 * mse(bl, f)               # 线性混合会有两个半透明方块,补偿后基本只有一个


def test_static_scene_is_identity():
    a = scene(100)
    out = interpolate(a, a, a, a, a, 0.5)
    assert mse(out, a) < 1.0


def test_structure_score_drops_on_ghosting():
    src = [scene(60)[..., 0], scene(120)[..., 0]]
    clean = [s.copy() for s in src]
    ghost = [np.clip(0.5 * scene(60)[..., 0] + 0.5 * scene(180)[..., 0], 0, 255).astype(np.uint8), src[1]]
    sc_clean, sc_ghost = structure_scores(clean, src), structure_scores(ghost, src)
    assert sc_clean[0] > 0.99 and sc_ghost[0] < sc_clean[0] - 0.02
