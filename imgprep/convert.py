"""通道 / 色彩空间转换模块.

所有函数接受 BGR 格式的输入 (OpenCV 默认格式)，
输出指定色彩空间的结果。
"""

from typing import Literal

import cv2
import numpy as np

__all__ = [
    "to_grayscale", "to_bgr", "to_rgb", "to_hsv", "to_lab",
    "convert_color",
]

# 支持的色彩空间枚举
ColorSpace = Literal["BGR", "RGB", "GRAY", "HSV", "LAB"]


def _validate_image(img: np.ndarray) -> None:
    if img.ndim not in (2, 3):
        raise ValueError(
            f"输入图像必须是 2D (灰度) 或 3D (彩色) 数组, 实际 ndim={img.ndim}"
        )
    if img.dtype != np.uint8:
        raise ValueError(f"输入图像 dtype 必须是 uint8, 实际 {img.dtype}")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """将 BGR 彩色图像转换为灰度图.

    Args:
        img: BGR 彩色图像, ndarray, shape=(H, W, 3).

    Returns:
        灰度图像, ndarray, shape=(H, W), dtype=uint8.

    Raises:
        ValueError: 输入不是 3 通道 BGR 图像.
    """
    _validate_image(img)
    if img.ndim == 2:
        return img.copy()
    if img.shape[2] != 3:
        raise ValueError(f"输入必须是 3 通道 BGR 图像, 实际通道数: {img.shape[2]}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """将 BGR 转换为 RGB.

    Args:
        img: BGR 图像, ndarray, shape=(H, W, 3).

    Returns:
        RGB 图像, ndarray, shape=(H, W, 3).
    """
    _validate_image(img)
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """将 RGB 转换为 BGR.

    Args:
        img: RGB 图像, ndarray, shape=(H, W, 3).

    Returns:
        BGR 图像, ndarray, shape=(H, W, 3).
    """
    _validate_image(img)
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_hsv(img: np.ndarray) -> np.ndarray:
    """将 BGR 图像转换为 HSV 色彩空间.

    Args:
        img: BGR 图像, ndarray, shape=(H, W, 3).

    Returns:
        HSV 图像, ndarray, shape=(H, W, 3).
    """
    _validate_image(img)
    if img.ndim == 2:
        raise ValueError("灰度图无法转换到 HSV.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_lab(img: np.ndarray) -> np.ndarray:
    """将 BGR 图像转换为 Lab 色彩空间.

    Args:
        img: BGR 图像, ndarray, shape=(H, W, 3).

    Returns:
        Lab 图像, ndarray, shape=(H, W, 3).
    """
    _validate_image(img)
    if img.ndim == 2:
        raise ValueError("灰度图无法转换到 Lab.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


# 色彩空间映射表
_COLOR_MAP: dict[tuple[ColorSpace, ColorSpace], int] = {
    ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
    ("BGR", "RGB"):  cv2.COLOR_BGR2RGB,
    ("BGR", "HSV"):  cv2.COLOR_BGR2HSV,
    ("BGR", "LAB"):  cv2.COLOR_BGR2LAB,
    ("RGB", "BGR"):  cv2.COLOR_RGB2BGR,
    ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
    ("RGB", "HSV"):  cv2.COLOR_RGB2HSV,
    ("RGB", "LAB"):  cv2.COLOR_RGB2LAB,
    ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
    ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
}


def convert_color(
    img: np.ndarray,
    src: ColorSpace,
    dst: ColorSpace,
) -> np.ndarray:
    """通用色彩空间转换.

    Args:
        img: 输入图像.
        src: 源色彩空间, 可选: "BGR", "RGB", "GRAY", "HSV", "LAB".
        dst: 目标色彩空间, 可选: "BGR", "RGB", "GRAY", "HSV", "LAB".

    Returns:
        转换后的图像.

    Raises:
        ValueError: 不支持的转换方向.
    """
    _validate_image(img)
    if src == dst:
        return img.copy()

    key = (src, dst)
    code = _COLOR_MAP.get(key)
    if code is None:
        raise ValueError(
            f"不支持的色彩空间转换: {src} -> {dst}. "
            f"支持的转换: {list(_COLOR_MAP.keys())}"
        )

    return cv2.cvtColor(img, code)
