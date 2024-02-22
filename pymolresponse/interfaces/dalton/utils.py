from pymolresponse.operators import Operator


def clean_dalton_label(original_label: str) -> str:
    """Operator/integral labels in DALTON are in uppercase and may have
    spaces in them; replace spaces with underscores and make all
    letters lowercase.

    >>> clean_dalton_label("PSO 002")
    'pso_002'
    """
    return original_label.lower().replace(" ", "_")


def dalton_label_to_operator(label: str) -> Operator:
    label = clean_dalton_label(label)

    coord1_to_slice = {"x": 0, "y": 1, "z": 2}
    coord2_to_slice = {
        "xx": 0,
        "xy": 1,
        "xz": 2,
        "yy": 3,
        "yz": 4,
        "zz": 5,
        "yx": 1,
        "zx": 2,
        "zy": 4,
    }
    slice_to_coord1 = {v: k for (k, v) in coord1_to_slice.items()}

    # dipole length
    if "diplen" in label:
        operator_label = "dipole"
        _coord = label[0]
        slice_idx = coord1_to_slice[_coord]
        is_imaginary = False
        is_spin_dependent = False
    # dipole velocity
    elif "dipvel" in label:
        operator_label = "dipvel"
        _coord = label[0]
        slice_idx = coord1_to_slice[_coord]
        is_imaginary = True
        is_spin_dependent = False
    # angular momentum
    elif "angmom" in label:
        operator_label = "angmom"
        _coord = label[0]
        slice_idx = coord1_to_slice[_coord]
        is_imaginary = True
        is_spin_dependent = False
    # spin-orbit
    elif "spnorb" in label:
        operator_label = "spinorb"
        _coord = label[0]
        slice_idx = coord1_to_slice[_coord]
        is_imaginary = True
        is_spin_dependent = True
        _nelec = label[1]
        if _nelec in ("1", "2"):
            operator_label += _nelec
        # combined one- and two-electron
        elif _nelec in (" ", "_"):
            operator_label += "c"
        else:
            pass
    # Fermi contact
    elif "fc" in label:
        operator_label = "fermi"
        _atomid = label[6 : 6 + 2]
        slice_idx = int(_atomid) - 1
        is_imaginary = False
        is_spin_dependent = True
    # spin-dipole
    elif "sd" in label:
        operator_label = "sd"
        _coord_atom = label[3 : 3 + 3]
        _coord = label[7]
        _atomid = (int(_coord_atom) - 1) // 3
        _coord_1 = (int(_coord_atom) - 1) % 3
        _coord_2 = slice_to_coord1[_coord_1] + _coord
        slice_idx = (6 * _atomid) + coord2_to_slice[_coord_2]
        is_imaginary = False
        is_spin_dependent = True
    # TODO SD+FC?
    # nucleus-orbit
    elif "pso" in label:
        operator_label = "pso"
        # TODO coord manipulation
        is_imaginary = True
        # TODO is this correct?
        is_spin_dependent = False
        # FIXME
        slice_idx = None
    else:
        operator_label = ""
        is_imaginary = None
        is_spin_dependent = None
        slice_idx = None

    operator = Operator(
        label=operator_label,
        is_imaginary=is_imaginary,
        is_spin_dependent=is_spin_dependent,
        slice_idx=slice_idx,
    )

    return operator
