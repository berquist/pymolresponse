# def test_hydrogen_atom_electric():

#     mol = molecules.hydrogen_atom_sto3g(5)
#     mol.build()

#     mf = pyscf.scf.uhf.UHF(mol)
#     mf.scf()

#     C = utils.fix_mocoeffs_shape(mf.mo_coeff)
#     E = utils.fix_moenergies_shape(mf.mo_energy)
#     occupations = utils.occupations_from_pyscf_mol(mol, C)

#     calculator = electric.Polarizability(mol, C, E, occupations, hamiltonian='rpa', spin='singlet', frequencies=[0.0])
#     calculator.form_operators()
#     # for hamiltonian in ('rpa', 'tda'):
#     #     for spin in ('singlet', 'triplet'):
#     #         calculator.run(hamiltonian=hamiltonian, spin=spin)
#     #         calculator.form_results()

#     return

if __name__ == "__main__":
    # test_hydrogen_atom_electric()
    pass
