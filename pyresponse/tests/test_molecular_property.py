import numpy as np

from pyresponse import molecular_property


def test_molecular_property():
    pyscfmol = None
    mocoeffs = np.array([])
    moenergies = np.array([])
    occupations = np.array([])
    cls = molecular_property.MolecularProperty(pyscfmol, mocoeffs, moenergies, occupations)
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
    return


def test_response_property():
    pyscfmol = None
    mocoeffs = np.zeros((1, 2, 2))
    moenergies = np.zeros((1, 2, 2))
    occupations = [0 for _ in range(4)]
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
    return


def test_transition_property():
    pyscfmol = None
    mocoeffs = np.zeros((1, 2, 2))
    moenergies = np.zeros((1, 2, 2))
    occupations = [0 for _ in range(4)]
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
    return


if __name__ == '__main__':
    test_molecular_property()
    # test_response_property()
    # test_transition_property()
