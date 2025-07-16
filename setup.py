from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("-e")]

setup(
    name="AresLM",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.11",
)
