"""imgprep - 图像预处理工具库.

提供常用的图像预处理功能，包括：
  - 图像读写 (io)
  - 通道/色彩空间转换 (convert)
  - 亮度增强 (enhance)
  - 椒盐噪声/高斯噪声 (noise)
  - 运动模糊 (blur)
"""

from .io import imread, imwrite
from .convert import (
    to_grayscale,
    to_bgr,
    to_rgb,
    to_hsv,
    to_lab,
    convert_color,
)
from .enhance import adjust_brightness
from .noise import salt_pepper_noise, gaussian_noise
from .blur import motion_blur

__all__ = [
    "imread", "imwrite",
    "to_grayscale", "to_bgr", "to_rgb", "to_hsv", "to_lab", "convert_color",
    "adjust_brightness",
    "salt_pepper_noise", "gaussian_noise",
    "motion_blur",
]

__version__ = "0.1.0"
