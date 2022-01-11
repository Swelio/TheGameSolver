import os

import setuptools

import project_package

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), "r"
) as description_file:
    long_description = description_file.read()

setuptools.setup(
    name=project_package.__package_name__,
    description=project_package.__description__,
    long_description=long_description,
    version=project_package.__version__,
    author=project_package.__author__,
    license=project_package.__license__,
    url=project_package.__url__,
    python_requires="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[],
)
