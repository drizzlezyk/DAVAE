from distutils.core import setup
from setuptools import setup, find_packages

install_requires = [
      'numpy'
      'scanpy',
      'tensorflow',
      'scipy',
      'pandas',
      'seaborn',
      'python-igraph',
      'louvain',
      'desc',
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