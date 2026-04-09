"""
KAIROSYN-1 Package Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="kairosyn",
    version="1.0.0",
    author="Dustin Groves",
    author_email="research@or4cl3.ai",
    description="A Recursive Multimodal Architecture for Epinoetic Artificial Consciousness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/or4cl3-ai-1/Kairosyn-1",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-cov>=5.0.0",
            "black>=24.8.0",
            "isort>=5.13.0",
            "mypy>=1.11.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "kairosyn-train=scripts.train_sft:main",
            "kairosyn-eval=scripts.evaluate:main",
            "kairosyn-infer=scripts.inference:main",
        ]
    },
)
