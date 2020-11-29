# -*- coding: UTF-8 -*-
"""
@author: huanghui
@file: setup.py
@date: 2020/11/13
"""
from setuptools import find_packages, setup

setup(
    name="textToy",
    version="0.0.2",
    author="huanghui",
    author_email="m130219330432163.com",
    description="A simple tool for pretrain model",
    license="Apache",
    url="https://github.com/huanghuidmml/textToy",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.6.0"
)
