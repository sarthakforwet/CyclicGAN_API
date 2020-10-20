from setuptools import setup
from setuptools import find_packages

with open("Readme.md") as f:
    long_description = f.read()

setup(
    name="cyclicgan",
    version="0.0.1",
    description="An API implementation of CyclicGAN network.",
    author="Sarthak Khandelwal",
    author_email="sarthakkhandelwal032000@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarthakforwet/CyclicGAN_Implementation",
    packages=find_packages(),
    python_requires= '>=3.6',
    )