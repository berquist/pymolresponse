from utils import dalton_label_to_operator


def run_dalton_label_to_operator(dalton_label, operator_label, slice_idx, is_imaginary, is_spin_dependent):
    operator = dalton_label_to_operator(dalton_label)
    assert operator.label == operator_label
    # TODO slice_idx
    assert operator.is_imaginary == is_imaginary
    assert operator.is_spin_dependent == is_spin_dependent
    return

def test_dalton_label_to_operator():

    run_dalton_label_to_operator("zdiplen", "dipole", 2, False, False)
    run_dalton_label_to_operator("ydipvel", "dipvel", 1, True, False)
    run_dalton_label_to_operator("xangmom", "angmom", 0, True, False)
    run_dalton_label_to_operator("sd_004_y", "sd", 7, False, True)
    run_dalton_label_to_operator("fc_h__02", "fermi", 1, False, True)
    run_dalton_label_to_operator("fc_cu_02", "fermi", 1, False, True)
    run_dalton_label_to_operator("fc_cu_87", "fermi", 86, False, True)
    run_dalton_label_to_operator("z1spnorb", "spinorb1", 2, True, True)
    # TODO 2-el, combined spin-orbit
    # TODO nucleus-orbit/pso

    pass
