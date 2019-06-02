#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

test_requirements = [
    'tox',
    'flake8==2.6.0'
]

setup(
    name='aisimplekit',
    version='0.0.1',
    description="Simple lib for various machine learning and AI tasks.",
    long_description=readme,
    author="Said Aitmbarek",
    author_email='said.aitmbarek@gmail.com',
    url='https://github.com/greyzor/aisimplekit',
    packages=find_packages(),
    # [
    #     'aisimplekit',
    # ],
    package_dir={'aisimplekit': 'aisimplekit'},
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='aisimplekit',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)