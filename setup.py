import setuptools


def setup_pyresponse():

    setuptools.setup(
        name="pyresponse",
        version="0.1alpha",
        packages=setuptools.find_packages(exclude=["*test*"]),

        install_requires=[
            "numpy", "scipy", "cclib", "periodictable",
        ],

        # metadata
        author="Eric Berquist",
        maintainer="Eric Berquist",
        # TODO read this and long_description from README.md
        description="Molecular frequency-dependent response properties for arbitrary operators",
        license="BSD 3-Clause License",
        url="https://github.com/berquist/pyresponse",
        project_urls={
            "Documentation": "https://berquist.github.io/pyresponse_docs/",
        },
    )


if __name__ == "__main__":
    setup_pyresponse()
