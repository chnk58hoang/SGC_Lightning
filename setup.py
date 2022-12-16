from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f]

setup(
    name='sgc',
    version='0.1.0',
    description='SGC implementation in PyTorch Lightning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/chnk58hoang/SGC_Lightning',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.7',
)
