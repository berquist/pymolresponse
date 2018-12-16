"""
pyresponse
Molecular frequency-dependent response properties for arbitrary operators
"""
from setuptools import setup, find_packages
import versioneer

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    name="pyresponse",
    author="Eric Berquist",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='BSD-3-Clause',
    packages=find_packages(exclude=["*test*"]),
    url="https://github.com/berquist/pyresponse",
    python_requires=">=3.6",
    # install_requires=[
    #     "numpy", "scipy", "cclib", "periodictable",
    # ],
    project_urls={
        "Documentation": "https://berquist.github.io/pyresponse_docs/",
    },
)
