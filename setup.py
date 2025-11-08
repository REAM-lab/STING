from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

setup(
        name='sting',
        version='0.1.0',
        author='sting_developers',
        author_email='your.email@example.com',
        description='Specialized tool for inverter-based grids',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        packages=find_packages(),
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: MacOS :: MacOS X',
            'Topic :: Scientific/Engineering',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        python_requires='>=3.14',
        install_requires=[
            "gamspy",
	    "importlib",
	    "pandas",
	    "numpy",
            "scipy",
            "tabulate"
        ],
    )
