import numpy as np
import pytest

from pymolresponse import molecular_property
from pymolresponse.core import Program


@pytest.mark.skip("TODO these have been turned into ABCs")
def test_molecular_property() -> None:
    pyscfmol = None
    mocoeffs = np.array([])
    moenergies = np.array([])
    occupations = np.array([])
    cls = molecular_property.MolecularProperty(
        Program.PySCF, pyscfmol, mocoeffs, moenergies, occupations
    )
    try:
        cls.form_operators()
    except NotImplementedError as e:
        message = e.args[0]
        assert message == "This must be implemented in a grandchild class."
    try:
        cls.form_results()
    except NotImplementedError as e:
        message = e.args[0]
        assert message == "This must be implemented in a grandchild class."


@pytest.mark.skip()
def test_response_property() -> None:
    pyscfmol = None  # noqa: F841
    mocoeffs = np.zeros((1, 2, 2))  # noqa: F841
    moenergies = np.zeros((1, 2, 2))  # noqa: F841
    occupations = [0 for _ in range(4)]  # noqa: F841
    # TODO Turns out invoking the solver does a lot automatically...
    # cls = molecular_property.ResponseProperty(pyscfmol, mocoeffs, moenergies, occupations)
    # try:
    #     cls.form_operators()
    # except NotImplementedError as e:
    #     message = e.args[0]
    #     assert message == "This must be implemented in a child class."
    # try:
    #     cls.form_results()
    # except NotImplementedError as e:
    #     message = e.args[0]
    #     assert message == "This must be implemented in a child class."


@pytest.mark.skip()
def test_transition_property() -> None:
    pyscfmol = None  # noqa: F841
    mocoeffs = np.zeros((1, 2, 2))  # noqa: F841
    moenergies = np.zeros((1, 2, 2))  # noqa: F841
    occupations = [0 for _ in range(4)]  # noqa: F841
    # TODO Turns out invoking the solver does a lot automatically...
    # cls = molecular_property.TransitionProperty(pyscfmol, mocoeffs, moenergies, occupations)
    # try:
    #     cls.form_operators()
    # except NotImplementedError as e:
    #     message = e.args[0]
    #     assert message == "This must be implemented in a child class."
    # try:
    #     cls.form_results()
    # except NotImplementedError as e:
    #     message = e.args[0]
    #     assert message == "This must be implemented in a child class."


if __name__ == "__main__":
    test_molecular_property()
    # test_response_property()
    # test_transition_property()
