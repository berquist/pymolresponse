from pymolresponse.tests.test_runners import run_dalton_label_to_operator


def test_dalton_label_to_operator() -> None:
    """Test that
    1. Operator attributes are properly set based on a DALTON label, and that
    2. __str__ is correct
    """

    operator = run_dalton_label_to_operator("zdiplen", "dipole", 2, False, False)
    assert (
        operator.__str__()
        == """Operator(label="dipole", is_imaginary=False, is_spin_dependent=False, triplet=False, slice_idx=2)"""
    )
    operator = run_dalton_label_to_operator("ydipvel", "dipvel", 1, True, False)
    assert (
        operator.__str__()
        == """Operator(label="dipvel", is_imaginary=True, is_spin_dependent=False, triplet=False, slice_idx=1)"""
    )
    operator = run_dalton_label_to_operator("xangmom", "angmom", 0, True, False)
    assert (
        operator.__str__()
        == """Operator(label="angmom", is_imaginary=True, is_spin_dependent=False, triplet=False, slice_idx=0)"""
    )
    operator = run_dalton_label_to_operator("sd_004_y", "sd", 7, False, True)
    assert (
        operator.__str__()
        == """Operator(label="sd", is_imaginary=False, is_spin_dependent=True, triplet=False, slice_idx=7)"""
    )
    operator = run_dalton_label_to_operator("fc_h__02", "fermi", 1, False, True)
    assert (
        operator.__str__()
        == """Operator(label="fermi", is_imaginary=False, is_spin_dependent=True, triplet=False, slice_idx=1)"""
    )
    operator = run_dalton_label_to_operator("fc_cu_02", "fermi", 1, False, True)
    assert (
        operator.__str__()
        == """Operator(label="fermi", is_imaginary=False, is_spin_dependent=True, triplet=False, slice_idx=1)"""
    )
    operator = run_dalton_label_to_operator("fc_cu_87", "fermi", 86, False, True)
    assert (
        operator.__str__()
        == """Operator(label="fermi", is_imaginary=False, is_spin_dependent=True, triplet=False, slice_idx=86)"""
    )
    operator = run_dalton_label_to_operator("z1spnorb", "spinorb1", 2, True, True)
    assert (
        operator.__str__()
        == """Operator(label="spinorb1", is_imaginary=True, is_spin_dependent=True, triplet=False, slice_idx=2)"""
    )

    # TODO 2-el, combined spin-orbit
    # TODO nucleus-orbit/pso
