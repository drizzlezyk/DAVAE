from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='davae',
    version='0.0.1',
    keywords=('integration', 'single cell'),
    description='Domain adaption variational autoencoder for integration of single cell data',
    license='MIT License',
    url='https://github.com/drizzlezyk/DAVAE',
    author='Yuanke Zhong',
    author_email='yuanke.zhong@nwpu-bioinformatics.com',
    packages=find_packages(),
    platforms='any',
)