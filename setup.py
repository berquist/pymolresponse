"""
pyresponse
Molecular frequency-dependent response properties for arbitrary operators
"""
from pathlib import Path
from setuptools import setup, find_packages
from typing import List, Sequence

import versioneer

short_description = __doc__.split("\n")

try:
    with open("README.md", "r", encoding="utf-8") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


def _get_files_recursive(top: Path) -> Sequence[Path]:
    def get_files_recursive_acc(top: Path, files: List[Path]) -> None:
        for f in top.iterdir():
            if f.is_file():
                files.append(f.resolve(strict=True))
            elif f.is_dir():
                get_files_recursive_acc(f.resolve(strict=True), files)
        return
    files: List[Path] = []
    get_files_recursive_acc(top, files)
    return files


def _get_package_data() -> Sequence[Path]:
    files: List[Path] = []
    datadir = Path(__file__).resolve().parent / "pyresponse" / "data"
    files.extend(_get_files_recursive(datadir / "coords"))
    files.extend(_get_files_recursive(datadir / "reference"))
    return sorted(files)


if __name__ == "__main__":

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
        package_data={"pyresponse": _get_package_data()},
        url="https://github.com/berquist/pyresponse",
        python_requires=">=3.6",
        # install_requires=[
        #     "numpy", "scipy", "cclib", "periodictable",
        # ],
        project_urls={
            "Documentation": "https://berquist.github.io/pyresponse_docs/",
        },
    )
