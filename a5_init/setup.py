from setuptools import setup, find_packages

setup(
    name="a5-init",
    version="1.0.0",
    author="Rakitin",
    description="A5 Symmetry Initialization for Neural Networks",
    packages=find_packages(),
    install_requires=["torch", "numpy"],
    python_requires=">=3.7",
)
