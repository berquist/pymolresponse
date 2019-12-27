"""
isort:skip_file
"""

# pylint: disable=unused-import

from . import (
    ao2mo,
    constants,
    cphf,
    ecd,
    electric,
    explicit_equations_full,
    explicit_equations_partial,
    helpers,
    integrals,
    iterators,
    magnetic,
    molecular_property,
    operators,
    optrot,
    td,
    utils,
)

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
