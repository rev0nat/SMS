#!/usr/bin/python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*-coding:Utf-8 -*

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='celestools',
    version='1.0.7',
    description='Tools to solve problems about orbital flight',
    long_description=readme,
    author='Alexis Petit',
    author_email='alexis.petit@sharemyspace.global',
    url='ssh://git@SMS2:/srv/git/celestools.git',
    license=license,
    packages=find_packages(exclude=('tests'))
)

