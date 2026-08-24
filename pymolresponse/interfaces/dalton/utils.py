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
        coord = label[0]
        slice_idx = coord1_to_slice[coord]
        is_imaginary = False
        is_spin_dependent = False
    # dipole velocity
    elif "dipvel" in label:
        operator_label = "dipvel"
        coord = label[0]
        slice_idx = coord1_to_slice[coord]
        is_imaginary = True
        is_spin_dependent = False
    # angular momentum
    elif "angmom" in label:
        operator_label = "angmom"
        coord = label[0]
        slice_idx = coord1_to_slice[coord]
        is_imaginary = True
        is_spin_dependent = False
    # spin-orbit
    elif "spnorb" in label:
        operator_label = "spinorb"
        coord = label[0]
        slice_idx = coord1_to_slice[coord]
        is_imaginary = True
        is_spin_dependent = True
        nelec = label[1]
        if nelec in {"1", "2"}:
            operator_label += nelec
        # combined one- and two-electron
        elif nelec in {" ", "_"}:
            operator_label += "c"
    # Fermi contact
    elif "fc" in label:
        operator_label = "fermi"
        atomid = label[6 : 6 + 2]
        slice_idx = int(atomid) - 1
        is_imaginary = False
        is_spin_dependent = True
    # spin-dipole
    elif "sd" in label:
        operator_label = "sd"
        coord_atom = label[3 : 3 + 3]
        coord = label[7]
        atomid = (int(coord_atom) - 1) // 3
        coord_1 = (int(coord_atom) - 1) % 3
        coord_2 = slice_to_coord1[coord_1] + coord
        slice_idx = (6 * atomid) + coord2_to_slice[coord_2]
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
        msg = f"Unhandled DALTON operator label: {label}"
        raise RuntimeError(msg)

    # TODO this hack should go away once the PSO label is fixed
    assert slice_idx is not None

    operator = Operator(
        label=operator_label,
        is_imaginary=is_imaginary,
        is_spin_dependent=is_spin_dependent,
        slice_idx=slice_idx,
        ao_integrals=None,  # ty: ignore[invalid-argument-type]
    )

    return operator
