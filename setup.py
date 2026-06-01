from setuptools import find_packages, setup

setup(
    name="imgprep",
    version="0.1.0",
    description="图像预处理工具库 —— 读图、通道转换、亮度增强、椒盐噪声、运动模糊",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="valen",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
