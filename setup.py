from setuptools import find_packages, setup

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name="deeplab",
    version="0.1.0",
    description="Deep Learning Lab Package",
    packages=find_packages(),
    include_package_data=True,)