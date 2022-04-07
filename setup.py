#!/usr/bin/env python
from setuptools import setup, find_packages

version = "0.0.1"

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "setuptools",
]

setup(
    name="rubiks-cube",
    version=version,
    description=("A Rubiks Cube game and RL project."),
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Emre Sevgen",
    author_email="sesevgen@gmail.com",
    url="https://github.com/sesevgen/rubiks-cube",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
)
