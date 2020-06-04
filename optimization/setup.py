# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:54:10 2018

@author: dominic
"""

from setuptools import setup
from setuptools import find_packages

from distutils.util import convert_path 
main_ns = {}
ver_path = convert_path('optimization/version.py')
with open(ver_path) as ver_file:
	exec(ver_file.read(), main_ns)


setup(
    name='optimization',
    version=main_ns['__version__'],
    packages=find_packages(),
    long_description=open('README.txt').read(),
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=['numpy','scipy'],
)

