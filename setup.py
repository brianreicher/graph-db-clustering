import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setuptools.setup(
    name="graphnn",
    version="0.1",
    author="Brian Reicher",
    author_email="reicher.b@northeastern.edu",
    description="Graph Neural Network implementation with Neo4j backend",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brianreicher/graph-nn",
    # packages=setuptools.find_packages(),
    package_dir={"": "graphnn"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements
)