from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='NumbaML',
    version='1.0.21',
    packages=find_packages(),
    url='https://github.com/jcatankard/NumbaML',
    author='Josh Tankard',
    description='Linear regression with Numba',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numba', 'numpy', 'scipy'],
)
