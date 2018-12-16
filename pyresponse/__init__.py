from . import ao2mo
from . import constants
from . import cphf
from . import ecd
from . import electric
from . import explicit_equations_full
from . import explicit_equations_partial
from . import helpers
from . import integrals
from . import iterators
from . import magnetic
from . import molecular_property
from . import operators
from . import optrot
from . import td
from . import utils

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
