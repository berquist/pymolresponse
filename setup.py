from pathlib import Path
from typing import List, Sequence

from setuptools import find_packages, setup

import versioneer


def _get_files_recursive(top: Path) -> Sequence[Path]:
    def get_files_recursive_acc(top: Path, files: List[Path]) -> None:
        for f in top.iterdir():
            if f.is_file():
                files.append(f.resolve(strict=True))
            elif f.is_dir():
                get_files_recursive_acc(f.resolve(strict=True), files)

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
        author="Eric Berquist",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        packages=find_packages(exclude=["*test*"]),
        package_data={"pyresponse": tuple(str(f) for f in _get_package_data())},
        project_urls={"Documentation": "https://berquist.github.io/pyresponse_docs/"},
    )
