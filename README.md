# imgprep — 图像预处理工具库

提供常用图像预处理功能的轻量 Python 库，基于 OpenCV + NumPy。

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 接口

| 模块 | 函数 | 功能 |
|------|------|------|
| `imgprep.io` | `imread(path)` | 读取图像 |
| `imgprep.io` | `imwrite(path, img)` | 保存图像 |
| `imgprep.convert` | `to_grayscale(img)` | BGR → 灰度 |
| `imgprep.convert` | `to_rgb(img)` | BGR → RGB |
| `imgprep.convert` | `to_hsv(img)` | BGR → HSV |
| `imgprep.convert` | `to_lab(img)` | BGR → Lab |
| `imgprep.convert` | `convert_color(img, src, dst)` | 通用色彩空间转换 |
| `imgprep.enhance` | `adjust_brightness(img, alpha, beta)` | 亮度 / 对比度调整 |
| `imgprep.noise` | `salt_pepper_noise(img, salt_prob, pepper_prob)` | 添加椒盐噪声 |
| `imgprep.blur` | `motion_blur(img, kernel_size, angle)` | 运动模糊 |

## 快速使用

```python
import cv2
from imgprep import imread, imwrite
from imgprep import to_grayscale, to_hsv
from imgprep import adjust_brightness
from imgprep import salt_pepper_noise
from imgprep import motion_blur

# 读图
img = imread("input.jpg")

# 通道转换
gray = to_grayscale(img)
hsv = to_hsv(img)

# 亮度增强
brighter = adjust_brightness(img, alpha=1.2, beta=30)

# 椒盐噪声
noisy = salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)

# 运动模糊 (45° 方向)
blurred = motion_blur(img, kernel_size=21, angle=45)

imwrite("output.jpg", blurred)
```

## 依赖

- Python ≥ 3.8
- numpy ≥ 1.21
- opencv-python ≥ 4.5

## 许可证

MIT
