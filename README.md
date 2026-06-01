# imgprep — 图像预处理工具库

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**imgprep** 是一个轻量级图像预处理 Python 库，基于 OpenCV + NumPy 实现，提供计算机视觉任务中常用的图像增强与预处理功能。

---

## 安装

### 方式一：通过 GitHub 安装（推荐）

```bash
pip install git+https://github.com/VanlenFrank/data-augmentation.git
```

### 方式二：本地开发安装

```bash
git clone https://github.com/VanlenFrank/data-augmentation.git
cd data-augmentation
pip install -e .
```

### 验证安装

```python
import imgprep
print(imgprep.__version__)  # 0.1.0
```

---

## 快速开始

```python
import numpy as np
from imgprep import imread, imwrite
from imgprep import to_grayscale, to_hsv
from imgprep import adjust_brightness
from imgprep import salt_pepper_noise, gaussian_noise
from imgprep import motion_blur

# 1. 读取图像
img = imread("input.jpg")                # shape=(H, W, 3), dtype=uint8, BGR 格式

# 2. 色彩空间转换
gray = to_grayscale(img)                 # 灰度化, shape=(H, W)
hsv  = to_hsv(img)                       # BGR → HSV, shape=(H, W, 3)

# 3. 亮度增强 (alpha=对比度, beta=亮度偏置)
brighter = adjust_brightness(img, alpha=1.2, beta=30)

# 4. 椒盐噪声
noisy_sp = salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01)

# 5. 高斯噪声
noisy_gs = gaussian_noise(img, mean=0.0, sigma=25.0)

# 6. 运动模糊 (水平方向, 核大小 21)
blurred = motion_blur(img, kernel_size=21, angle=0)

# 7. 保存结果
imwrite("output.jpg", blurred)
```

---

## 完整 API 文档

### `imgprep.io` — 图像读写

#### `imread(path, flags=cv2.IMREAD_COLOR)`

读取图像文件。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | `str \| Path` | — | 图像文件路径，支持 `.jpg/.png/.bmp/.tiff` 等 OpenCV 支持的格式 |
| `flags` | `int` | `cv2.IMREAD_COLOR` | 读取模式：`IMREAD_COLOR`(BGR 三通道)、`IMREAD_GRAYSCALE`(单通道)、`IMREAD_UNCHANGED`(含 alpha) |

| 返回 | 说明 |
|------|------|
| `np.ndarray` | shape=(H, W, 3) 或 (H, W)，dtype=uint8 |

**异常：**
- `FileNotFoundError` — 文件不存在
- `ValueError` — 文件损坏或格式不支持

**示例：**
```python
color = imread("photo.jpg")                    # 默认彩色
gray  = imread("photo.jpg", cv2.IMREAD_GRAYSCALE)  # 灰度读取
```

---

#### `imwrite(path, img)`

保存图像到文件，格式由扩展名自动推断。

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str \| Path` | 输出路径（父目录会自动创建） |
| `img` | `np.ndarray` | 图像数据 |

**异常：**
- `ValueError` — 图像数据为空或保存失败

**示例：**
```python
imwrite("result/processed.png", img)
```

---

### `imgprep.convert` — 通道 / 色彩空间转换

> **注意：** 输入一律采用 OpenCV 默认的 BGR 格式。

#### `to_grayscale(img)`

BGR 彩色 → 灰度图。

| 参数 | 类型 | 说明 |
|------|------|------|
| `img` | `np.ndarray` | BGR 图像，shape=(H, W, 3) |

**返回：** `np.ndarray`，shape=(H, W)，dtype=uint8

```python
gray = to_grayscale(img)
```

---

#### `to_rgb(img)`

BGR → RGB（常用于 matplotlib 显示）。

```python
rgb = to_rgb(img)
# plt.imshow(rgb)  # matplotlib 需要 RGB 顺序
```

#### `to_bgr(img)`

RGB → BGR。

```python
bgr = to_bgr(rgb_img)
```

#### `to_hsv(img)`

BGR → HSV 色彩空间。

```python
hsv = to_hsv(img)
```

#### `to_lab(img)`

BGR → CIE Lab 色彩空间。

```python
lab = to_lab(img)
```

#### `convert_color(img, src, dst)`

通用色彩空间转换。

| 参数 | 类型 | 说明 |
|------|------|------|
| `img` | `np.ndarray` | 输入图像 |
| `src` | `str` | 源空间：`"BGR"` / `"RGB"` / `"GRAY"` / `"HSV"` / `"LAB"` |
| `dst` | `str` | 目标空间：同上 |

```python
# BGR → HSV
hsv = convert_color(img, "BGR", "HSV")

# GRAY → RGB
rgb = convert_color(gray_img, "GRAY", "RGB")
```

---

### `imgprep.enhance` — 亮度与对比度调整

#### `adjust_brightness(img, alpha=1.0, beta=0)`

线性亮度 / 对比度调整。

公式：`output = saturate(alpha × input + beta)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `img` | `np.ndarray` | — | 输入图像，uint8 |
| `alpha` | `float` | `1.0` | 对比度增益：>1 增强，<1 减弱 |
| `beta` | `int` | `0` | 亮度偏置：>0 变亮，<0 变暗 |

```python
# 提高亮度和对比度
enhanced = adjust_brightness(img, alpha=1.3, beta=40)

# 降低亮度
dark = adjust_brightness(img, alpha=0.7, beta=0)
```

---

### `imgprep.noise` — 噪声添加

#### `salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01, salt_val=255, pepper_val=0)`

随机将部分像素置为纯白（盐）或纯黑（椒）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `img` | `np.ndarray` | — | 输入图像，支持彩色和灰度 |
| `salt_prob` | `float` | `0.01` | 盐噪声概率 [0, 1) |
| `pepper_prob` | `float` | `0.01` | 椒噪声概率 [0, 1) |
| `salt_val` | `int` | `255` | 盐噪声像素值 |
| `pepper_val` | `int` | `0` | 椒噪声像素值 |

```python
# 5% 椒盐噪声
noisy = salt_pepper_noise(img, 0.025, 0.025)
```

---

#### `gaussian_noise(img, mean=0.0, sigma=25.0)`

添加服从 N(mean, σ²) 分布的高斯噪声。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `img` | `np.ndarray` | — | 输入图像，支持彩色和灰度 |
| `mean` | `float` | `0.0` | 高斯噪声均值 |
| `sigma` | `float` | `25.0` | 高斯噪声标准差（越大噪声越强） |

```python
# 轻微高斯噪声
noisy_light = gaussian_noise(img, sigma=10)

# 强烈高斯噪声
noisy_heavy = gaussian_noise(img, sigma=50)
```

---

### `imgprep.blur` — 运动模糊

#### `motion_blur(img, kernel_size=15, angle=0.0)`

通过沿指定方向的线性卷积核模拟运动模糊。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `img` | `np.ndarray` | — | 输入图像，支持彩色和灰度 |
| `kernel_size` | `int` | `15` | 卷积核大小（>=3 的奇数），越大轨迹越长 |
| `angle` | `float` | `0.0` | 运动方向角度（度）：0°=水平，90°=垂直 |

```python
# 水平运动模糊
blur_h = motion_blur(img, kernel_size=21, angle=0)

# 垂直运动模糊
blur_v = motion_blur(img, kernel_size=21, angle=90)

# 45° 方向
blur_d = motion_blur(img, kernel_size=15, angle=45)
```

---

## 完整示例：数据扩增管道

```python
from imgprep import imread, imwrite
from imgprep import adjust_brightness
from imgprep import salt_pepper_noise, gaussian_noise
from imgprep import motion_blur

img = imread("input.jpg")
base_name = "augmented"

# 生成 5 种扩增版本
augmentations = {
    "bright":      adjust_brightness(img, alpha=1.2, beta=30),
    "dark":        adjust_brightness(img, alpha=0.6, beta=0),
    "salt_pepper": salt_pepper_noise(img, 0.02, 0.02),
    "gaussian":    gaussian_noise(img, sigma=30),
    "motion_blur": motion_blur(img, kernel_size=21, angle=0),
}

for name, aug_img in augmentations.items():
    imwrite(f"{base_name}_{name}.jpg", aug_img)
    print(f"Saved: {base_name}_{name}.jpg")
```

---

## 本地开发

```bash
# 克隆
git clone https://github.com/VanlenFrank/data-augmentation.git
cd data-augmentation

# 安装开发依赖
pip install -e .
pip install pytest

# 运行测试
pytest tests/ -v

# 运行示例
python examples/demo.py
```

## 测试

项目包含 31 项单元测试，覆盖正常使用、边界条件和异常输入：

```bash
pytest tests/ -v
```

## 依赖

- Python ≥ 3.8
- numpy ≥ 1.21
- opencv-python ≥ 4.5

## 许可证

MIT

## 项目地址

[https://github.com/VanlenFrank/data-augmentation](https://github.com/VanlenFrank/data-augmentation)
