"""亮度增强模块."""

import cv2
import numpy as np

__all__ = ["adjust_brightness"]


def adjust_brightness(
    img: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0,
) -> np.ndarray:
    """调整图像亮度与对比度.

    使用线性变换: output = alpha * input + beta.
      - alpha > 1.0 增强对比度; 0 < alpha < 1 降低对比度.
      - beta  > 0   增加亮度;    beta < 0    降低亮度.

    Args:
        img:   输入图像, uint8, ndarray.
        alpha: 对比度增益, 默认 1.0 (不变).
        beta:  亮度偏置,   默认 0   (不变).

    Returns:
        调整后的图像, 值域已被裁剪到 [0, 255].

    Raises:
        ValueError: img 为空或参数无效.
    """
    if img.size == 0:
        raise ValueError("输入图像为空.")
    if alpha < 0:
        raise ValueError(f"alpha 不能为负, 当前: {alpha}")

    # OpenCV 的 convertScaleAbs 做 out = saturate(alpha * img + beta)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted
