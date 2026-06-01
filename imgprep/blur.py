"""运动模糊模块."""

import cv2
import numpy as np

__all__ = ["motion_blur"]


def _create_motion_kernel(kernel_size: int, angle: float) -> np.ndarray:
    """创建运动模糊卷积核.

    Args:
        kernel_size: 核大小 (像素), 值越大模糊越重.
        angle:       运动方向角度 (度), 0° 表示水平向右, 90° 表示垂直向下.

    Returns:
        归一化的运动模糊核, ndarray, shape=(kernel_size, kernel_size).
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # 将角度转为弧度
    theta = np.deg2rad(angle)

    for i in range(kernel_size):
        # 沿运动方向等距采点
        offset = i - center
        x = center + int(round(offset * np.cos(theta)))
        y = center + int(round(offset * np.sin(theta)))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    # 归一化
    s = kernel.sum()
    if s > 0:
        kernel /= s
    else:
        kernel[center, center] = 1.0

    return kernel


def motion_blur(
    img: np.ndarray,
    kernel_size: int = 15,
    angle: float = 0.0,
) -> np.ndarray:
    """对图像施加运动模糊.

    通过沿指定方向的线性卷积核模拟相机 / 物体运动产生的模糊效果.

    Args:
        img:         输入图像, uint8, ndarray.
        kernel_size: 卷积核大小 (>=3 的奇数), 值越大运动轨迹越长, 默认 15.
        angle:       运动方向角度 (度), 默认 0° (水平).

    Returns:
        运动模糊后的图像, ndarray, dtype=uint8.

    Raises:
        ValueError: kernel_size 不是 >=3 的奇数; 或 img 为空.
    """
    if img.size == 0:
        raise ValueError("输入图像为空.")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size 必须是 >=3 的奇数, 当前: {kernel_size}"
        )

    kernel = _create_motion_kernel(kernel_size, angle)

    # 用 filter2D 做卷积
    blurred = cv2.filter2D(img, -1, kernel)

    return blurred
