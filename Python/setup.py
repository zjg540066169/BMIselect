#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:30:39 2022

@author: Jungang Zou
"""

#!/usr/bin/env python
# coding: utf-8

from setuptools import setup
from setuptools import find_packages

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
    
VERSION = '0.0.9'

setup(
    name='bmiselect',  # package name
    version=VERSION,  # package version
    author="Jungang Zou",
    author_email="jungang.zou@gmail.com",
    url="https://github.com/zjg540066169/Bmiselect",
    description='Bayesian MI-LASSO for variable selection on multiply-imputed data. ' ,  # package description
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Information Analysis",
        ],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    project_urls={
        "Code":"https://github.com/zjg540066169/Bmiselect",
        },
    python_requires=">=3.7.*",
    install_requires=[
        "pymc3>=3.11.5",
        "theano-pymc>=1.1.2",
        "mkl>=2.4.0",
        "mkl-service>=2.4.0",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "pandas",
        "seaborn",
        "arviz",
        "xarray",
        "statsmodels"
        ],
    
    
)