from distutils.core import setup
from setuptools import setup, find_packages

install_requires = [
      'numpy==1.18.5',
      'scanpy==1.5.1',
      'tensorflow==2.3.0',
      'scipy==1.4.1',
      'pandas==1.1.0',
      'python-igraph==0.8.2',
      'louvain==0.7.0',
      'desc==2.1.1',
]

setup(
    name='davae',
    version='0.0.3',
    keywords=('integration', 'single cell'),
    description='Domain adaption variational autoencoder for integration of single cell data',
    license='MIT License',
    url='https://github.com/drizzlezyk/DAVAE',
    author='Yuanke Zhong',
    author_email='yuanke.zhong@nwpu-bioinformatics.com',
    packages=find_packages(),
    platforms='any',
    install_requires=install_requires,
)