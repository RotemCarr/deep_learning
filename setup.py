"""Setup module for packaging."""
from setuptools import setup, find_packages

setup(name="deep_learning",
      version="0.1.0",
      packages=find_packages(include=["deep_learning.*"]),
      install_requires=[
            "numpy",
            "pandas",
            "setuptools",
            "pylint"]
      )
