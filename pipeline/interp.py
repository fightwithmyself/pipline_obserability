"""运动补偿补帧(纯 numpy,无需 GPU)。

要解决的问题:两张风格化关键帧之间隔了好几帧,直接线性混合会让运动的人物出现"双影"。
思路:**源视频的每一帧我们都有**,所以用源帧估计运动,再把风格化关键帧"搬"到中间时刻:

    1. 对中间帧 f 的源图,分别向前后两个关键帧 a、b 的源图做块匹配,得到稠密位移场(由粗到细三层金字塔);
    2. 按位移场把风格化的 a、b 回拉(backward warp)到 f 时刻;
    3. 用"回拉后的源图与 f 的源图的差"当可信度,遮挡/匹配失败的一侧权重自动降低,再按时间距离加权融合。

局限(要诚实写下来):块匹配只能描述平移,**旋转/形变/遮挡显露**的地方仍会有瑕疵;
风格化关键帧彼此细节不一致(模型独立生成)的部分,搬运并不能让它们一致。
"""
from __future__ import annotations

import numpy as np


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


def _down(g: np.ndarray, k: int) -> np.ndarray:
    """k 倍面积下采样(k 为 2 的幂时反复 2x2 平均;尺寸不整除时裁掉边)。"""
    while k > 1:
        h, w = g.shape[0] // 2 * 2, g.shape[1] // 2 * 2
        g = g[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        k //= 2
    return g


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """半径 r 的均值滤波(积分图,边缘复制)。"""
    if r <= 0:
        return a
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k)).astype(np.float32)


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """双线性采样,坐标越界时夹到边缘。img: HxW 或 HxWxC。"""
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1.001)
    ys = np.clip(ys, 0, h - 1.001)
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    fx, fy = (xs - x0).astype(np.float32), (ys - y0).astype(np.float32)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def estimate_flow(tgt: np.ndarray, ref: np.ndarray, *, levels: tuple[int, ...] = (8, 4, 2),
                  radius: tuple[int, ...] = (6, 2, 1), block: int = 4, smooth: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    """块匹配光流(由粗到细)。返回 (dx, dy, scale):在 1/scale 分辨率下,
    目标图 tgt 的像素 (x, y) 对应参考图 ref 的 (x+dx, y+dy)。tgt/ref 为 HxW 灰度 float32(全分辨率)。"""
    dx = dy = None
    prev = None
    g_t = g_r = None
    for lv, r in zip(levels, radius):
        t, f = _down(tgt, lv), _down(ref, lv)
        h, w = t.shape
        if dx is None:
            dx = np.zeros((h, w), np.float32)
            dy = np.zeros((h, w), np.float32)
        else:                                    # 上一层(更粗)的位移场放大到本层
            ph, pw = dx.shape
            sx, sy = (pw - 1) / max(w - 1, 1), (ph - 1) / max(h - 1, 1)
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32) * sx, np.arange(h, dtype=np.float32) * sy)
            ratio = prev / lv
            dx = _sample(dx, gx, gy) * ratio
            dy = _sample(dy, gx, gy) * ratio
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        best = np.full((h, w), np.inf, np.float32)
        bdx, bdy = dx.copy(), dy.copy()
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                cx, cy = dx + ox, dy + oy
                cost = np.abs(_sample(f, gx + cx, gy + cy) - t)
                cost = _box(cost, block) + 0.02 * (abs(ox) + abs(oy))     # 轻微偏好小位移,抑制纹理平坦处的乱跳
                m = cost < best
                best = np.where(m, cost, best)
                bdx = np.where(m, cx, bdx)
                bdy = np.where(m, cy, bdy)
        dx, dy = _box(bdx, smooth), _box(bdy, smooth)                     # 轻度平滑,避免块状边缘(过度平滑会把运动物体的位移抹向背景)
        prev = lv
    return dx, dy, prev


def warp(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, scale: int) -> np.ndarray:
    """按 (dx, dy)(1/scale 分辨率)把 img 回拉到目标视角;img 为 HxWxC 全分辨率。"""
    h, w = img.shape[:2]
    gh, gw = dx.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ux = _sample(dx, gx / scale, gy / scale) * scale
    uy = _sample(dy, gx / scale, gy / scale) * scale
    return _sample(img.astype(np.float32), gx + ux, gy + uy)


def interpolate(styled_a: np.ndarray, styled_b: np.ndarray, src_a: np.ndarray, src_b: np.ndarray,
                src_f: np.ndarray, t: float, *, sigma: float = 10.0) -> np.ndarray:
    """生成时刻 t(0..1,相对 a→b)的风格化中间帧。src_* 为同尺寸 RGB 源帧(uint8),styled_* 为风格化关键帧。"""
    ga, gb, gf = _gray(src_a), _gray(src_b), _gray(src_f)
    fa = estimate_flow(gf, ga)
    fb = estimate_flow(gf, gb)
    wa = warp(styled_a, *fa)
    wb = warp(styled_b, *fb)
    # 可信度:把源关键帧搬过来后,与真实的中间源帧越接近越可信
    ea = _box(np.abs(warp(src_a, *fa).mean(axis=2) - gf), 3)
    eb = _box(np.abs(warp(src_b, *fb).mean(axis=2) - gf), 3)
    ca = (1.0 - t) * np.exp(-ea / sigma)
    cb = t * np.exp(-eb / sigma)
    s = ca + cb + 1e-6
    out = (wa * (ca / s)[..., None] + wb * (cb / s)[..., None])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """旧方法:线性混合(基线)。"""
    return np.clip(np.rint(a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t), 0, 255).astype(np.uint8)
