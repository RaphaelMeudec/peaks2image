from setuptools import find_packages
from setuptools import setup


setup(
    name="peaks2image",
    version="0.1.0",
    url="https://github.com/raphaelmeudec/peaks2image",
    license='MIT',

    author="Raphael Meudec",
    author_email="raphael.meudec@gmail.com",

    description="A package for statistical image reconstruction from peak coordinates",
    long_description=open('README.md').read(),

    packages=find_packages(exclude=('tests',)),
    package_data={'peaks2image': ['model/assets/peaks2image.pt']},

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
