from setuptools import find_packages, setup


setup(
    name="datalib",
    version="0.2.1",
    author="Shane",
    author_email="shane@acryl.ai",
    description="A Python library for data processing and analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://rtl.acryl.ai/proj-skinex/burn-dataset",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.24.3",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "Pillow>=9.4.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
