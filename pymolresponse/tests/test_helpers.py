import numpy as np

from pymolresponse.helpers import calc_center_of_mass, nuclear_dipole_contribution


# data/reference/BC2H4_cation/orca_singlet_tda.out
#
# The origin for moment calculation is the CENTER OF MASS = ( 0.045689,
# 0.682354 -1.382979)
#
# -------------
# DIPOLE MOMENT
# -------------
#                                 X             Y             Z
# Electronic contribution:      0.99725      -1.16598       0.21107
# Nuclear contribution   :     -1.07799       1.14340      -0.03516
#                         -----------------------------------------
# Total Dipole Moment    :     -0.08073      -0.02257       0.17591
#                         -----------------------------------------
# Magnitude (a.u.)       :      0.19486
# Magnitude (Debye)      :      0.49530


def test_calc_center_of_mass() -> None:
    coords = np.asarray(
        [
            [-0.001566, -0.000849, -0.000612],
            [0.001024, -0.001041, 2.056253],
            [1.438845, 0.000172, -2.343696],
            [2.930559, -0.614282, -3.908508],
            [-1.072895, 1.832376, -1.829781],
            [-1.282679, 3.829018, -1.283344],
            [-2.514878, 1.269127, -3.241290],
        ]
    )
    # These are averaged, not isotopic
    masses = np.asarray([12.011, 1.008, 10.810, 1.008, 12.011, 1.008, 1.008])

    center_of_mass = calc_center_of_mass(coords, masses)
    np.testing.assert_allclose(
        center_of_mass, [0.045689, 0.682354, -1.382979], rtol=0.0, atol=1.0e-6
    )


def test_nuclear_dipole_contribution() -> None:
    coords = np.asarray(
        [
            [-0.001566, -0.000849, -0.000612],
            [0.001024, -0.001041, 2.056253],
            [1.438845, 0.000172, -2.343696],
            [2.930559, -0.614282, -3.908508],
            [-1.072895, 1.832376, -1.829781],
            [-1.282679, 3.829018, -1.283344],
            [-2.514878, 1.269127, -3.241290],
        ]
    )
    charges = np.asarray([6.0, 1.0, 5.0, 1.0, 6.0, 1.0, 1.0])
    masses = np.asarray([12.011, 1.008, 10.810, 1.008, 12.011, 1.008, 1.008])
    origin = calc_center_of_mass(coords, masses)

    nuclear_dipole = nuclear_dipole_contribution(coords, charges, origin)
    np.testing.assert_allclose(nuclear_dipole, [-1.07799, 1.14340, -0.03516], rtol=0.0, atol=1.0e-5)
