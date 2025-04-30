import numpy as np

from cgback.parser import pdb_atom_record_is_ca, system_from_pdb_path, system_from_cif_path, RESIDUE_TYPE_ENCODER, HEAVY_ATOM_TYPE_ENCODER, HYDROGEN_ATOM_TYPE_ENCODER, LOC_TYPE_ENCODER, BACKBONE_ATOM_TYPE_ENCODER


def test_pdb_atom_record_is_ca():
    pdb_atom_record = "ATOM      2  CA  SER A   1     -44.471   1.814   9.610  1.00  0.00           C"
    assert pdb_atom_record_is_ca(pdb_atom_record) == True

    pdb_atom_record = "ATOM      6  OG  SER A   1     -43.181   0.536  11.183  1.00  0.00           O"
    assert pdb_atom_record_is_ca(pdb_atom_record) == False


def test_system_from_pdb_file(tmp_path):
    # Create a temporary PDB file
    tmp_file = tmp_path / "test.pdb"
    tmp_file.write_text(
        "ATOM      1  N   SER A   1     -44.926   2.734  10.697  1.00  0.00           N  \n"
        "ATOM      2  CA  SER A   1     -44.471   1.814   9.610  1.00  0.00           C  \n"
        "ATOM      3  C   SER A   1     -45.511   1.713   8.489  1.00  0.00           C  \n"
        "ATOM      4  O   SER A   1     -46.707   1.747   8.768  1.00  0.00           O  \n"
        "ATOM      5  H   SER A   1     -45.804   2.417  11.094  1.00  0.00           H  \n"
        "ATOM      6  N   THR A   2     -36.145 -16.604  -6.518  1.00  0.00           N  \n"
        "ATOM      7  CA  THR A   2     -36.462 -15.953  -5.233  1.00  0.00           C  \n"
        "ATOM      8  C   THR A   2     -37.915 -15.465  -5.246  1.00  0.00           C  \n"
        "ATOM      9  O   THR A   2     -38.451 -15.183  -6.318  1.00  0.00           O  \n"
        "ATOM     10  H   THR A   2     -36.635 -16.243  -7.334  1.00  0.00           H  \n"
        "ATOM     11  N   PHE A   3     -40.964 -13.073  -2.428  1.00  0.00           N  \n"
        "ATOM     12  CA  PHE A   3     -41.975 -12.900  -1.374  1.00  0.00           C  \n"
        "ATOM     13  C   PHE A   3     -43.071 -11.881  -1.781  1.00  0.00           C  \n"
        "ATOM     14  O   PHE A   3     -42.740 -10.835  -2.391  1.00  0.00           O  \n"
        "ATOM     15  H   PHE A   3     -41.190 -12.602  -3.296  1.00  0.00           H  \n"
        "ATOM      1  N   ALA B   1     -44.926   2.734  10.697  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA B   1     -44.471   1.814   9.610  1.00  0.00           C  \n"
        "ATOM      3  C   ALA B   1     -45.511   1.713   8.489  1.00  0.00           C  \n"
        "ATOM      4  O   ALA B   1     -46.707   1.747   8.768  1.00  0.00           O  \n"
        "ATOM      5  H   ALA B   1     -45.804   2.417  11.094  1.00  0.00           H  \n"
        "ATOM      6  N   PRO B   2     -36.145 -16.604  -6.518  1.00  0.00           N  \n"
        "ATOM      7  CA  PRO B   2     -36.462 -15.953  -5.233  1.00  0.00           C  \n"
        "ATOM      8  C   PRO B   2     -37.915 -15.465  -5.246  1.00  0.00           C  \n"
        "ATOM      9  O   PRO B   2     -38.451 -15.183  -6.318  1.00  0.00           O  \n"
        "ATOM     10  H   PRO B   2     -36.635 -16.243  -7.334  1.00  0.00           H  \n"
        "ATOM     11  HD3 PRO B   2     -46.213 -29.701 -29.444  1.00  0.00           H  \n"
        "ATOM     12  N   CYS B   3     -40.964 -13.073  -2.428  1.00  0.00           N  \n"
        "ATOM     13  CA  CYS B   3     -41.975 -12.900  -1.374  1.00  0.00           C  \n"
        "ATOM     14  C   CYS B   3     -43.071 -11.881  -1.781  1.00  0.00           C  \n"
        "ATOM     15  O   CYS B   3     -42.740 -10.835  -2.391  1.00  0.00           O  \n"
        "ATOM     16  H   CYS B   3     -41.190 -12.602  -3.296  1.00  0.00           H  \n"
    )

    # Initialize the system object
    system = system_from_pdb_path(tmp_file)

    # Assert correct initialization
    assert system.chain_ids.shape == (6,)
    assert system.residue_types.shape == (6,)
    assert system.loc_types.shape == (6,)
    assert system.heavy_coordinates.shape == (6, 15, 3)
    assert system.hydrogen_coordinates.shape == (6, 16, 3)
    assert system.heavy_mask.shape == (6,)
    assert system.hydrogen_mask.shape == (6,)

    expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32)
    assert np.array_equal(system.chain_ids, expected)

    expected = np.array([RESIDUE_TYPE_ENCODER["SER"], RESIDUE_TYPE_ENCODER["THR"], RESIDUE_TYPE_ENCODER["PHE"], RESIDUE_TYPE_ENCODER["ALA"], RESIDUE_TYPE_ENCODER["PRO"], RESIDUE_TYPE_ENCODER["CYS"]], dtype=np.uint8)
    assert np.array_equal(system.residue_types, expected)

    expected = np.array([LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"], LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"]], dtype=np.uint8)
    assert np.array_equal(system.loc_types, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.heavy_mask, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.hydrogen_mask, expected)

    expected = np.array([-44.471, 1.814, 9.610], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[0, HEAVY_ATOM_TYPE_ENCODER["SER"]["CA"]], expected)
    expected = np.array([-36.462, -15.953, -5.233], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[1, HEAVY_ATOM_TYPE_ENCODER["THR"]["CA"]], expected)
    expected = np.array([-41.975, -12.900, -1.374], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[2, HEAVY_ATOM_TYPE_ENCODER["PHE"]["CA"]], expected)


def test_system_from_pdb_file_with_ter(tmp_path):
    # Create a temporary PDB file
    tmp_file = tmp_path / "test.pdb"
    tmp_file.write_text(
        "ATOM      1  N   SER A   1     -44.926   2.734  10.697  1.00  0.00           N  \n"
        "ATOM      2  CA  SER A   1     -44.471   1.814   9.610  1.00  0.00           C  \n"
        "ATOM      3  C   SER A   1     -45.511   1.713   8.489  1.00  0.00           C  \n"
        "ATOM      4  O   SER A   1     -46.707   1.747   8.768  1.00  0.00           O  \n"
        "ATOM      5  H   SER A   1     -45.804   2.417  11.094  1.00  0.00           H  \n"
        "ATOM      6  N   THR A   2     -36.145 -16.604  -6.518  1.00  0.00           N  \n"
        "ATOM      7  CA  THR A   2     -36.462 -15.953  -5.233  1.00  0.00           C  \n"
        "ATOM      8  C   THR A   2     -37.915 -15.465  -5.246  1.00  0.00           C  \n"
        "ATOM      9  O   THR A   2     -38.451 -15.183  -6.318  1.00  0.00           O  \n"
        "ATOM     10  H   THR A   2     -36.635 -16.243  -7.334  1.00  0.00           H  \n"
        "ATOM     11  N   PHE A   3     -40.964 -13.073  -2.428  1.00  0.00           N  \n"
        "ATOM     12  CA  PHE A   3     -41.975 -12.900  -1.374  1.00  0.00           C  \n"
        "ATOM     13  C   PHE A   3     -43.071 -11.881  -1.781  1.00  0.00           C  \n"
        "ATOM     14  O   PHE A   3     -42.740 -10.835  -2.391  1.00  0.00           O  \n"
        "ATOM     15  H   PHE A   3     -41.190 -12.602  -3.296  1.00  0.00           H  \n"
        "TER\n"
        "ATOM      1  N   ALA B   1     -44.926   2.734  10.697  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA B   1     -44.471   1.814   9.610  1.00  0.00           C  \n"
        "ATOM      3  C   ALA B   1     -45.511   1.713   8.489  1.00  0.00           C  \n"
        "ATOM      4  O   ALA B   1     -46.707   1.747   8.768  1.00  0.00           O  \n"
        "ATOM      5  H   ALA B   1     -45.804   2.417  11.094  1.00  0.00           H  \n"
        "ATOM      6  N   PRO B   2     -36.145 -16.604  -6.518  1.00  0.00           N  \n"
        "ATOM      7  CA  PRO B   2     -36.462 -15.953  -5.233  1.00  0.00           C  \n"
        "ATOM      8  C   PRO B   2     -37.915 -15.465  -5.246  1.00  0.00           C  \n"
        "ATOM      9  O   PRO B   2     -38.451 -15.183  -6.318  1.00  0.00           O  \n"
        "ATOM     10  H   PRO B   2     -36.635 -16.243  -7.334  1.00  0.00           H  \n"
        "ATOM     11  HD3 PRO B   2     -46.213 -29.701 -29.444  1.00  0.00           H  \n"
        "ATOM     12  N   CYS B   3     -40.964 -13.073  -2.428  1.00  0.00           N  \n"
        "ATOM     13  CA  CYS B   3     -41.975 -12.900  -1.374  1.00  0.00           C  \n"
        "ATOM     14  C   CYS B   3     -43.071 -11.881  -1.781  1.00  0.00           C  \n"
        "ATOM     15  O   CYS B   3     -42.740 -10.835  -2.391  1.00  0.00           O  \n"
        "ATOM     16  H   CYS B   3     -41.190 -12.602  -3.296  1.00  0.00           H  \n"
        "TER\n"
    )

    # Initialize the system object
    system = system_from_pdb_path(tmp_file)

    # Assert correct initialization
    assert system.chain_ids.shape == (6,)
    assert system.residue_types.shape == (6,)
    assert system.loc_types.shape == (6,)
    assert system.heavy_coordinates.shape == (6, 15, 3)
    assert system.hydrogen_coordinates.shape == (6, 16, 3)
    assert system.heavy_mask.shape == (6,)
    assert system.hydrogen_mask.shape == (6,)

    expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32)
    assert np.array_equal(system.chain_ids, expected)

    expected = np.array([RESIDUE_TYPE_ENCODER["SER"], RESIDUE_TYPE_ENCODER["THR"], RESIDUE_TYPE_ENCODER["PHE"], RESIDUE_TYPE_ENCODER["ALA"], RESIDUE_TYPE_ENCODER["PRO"], RESIDUE_TYPE_ENCODER["CYS"]], dtype=np.uint8)
    assert np.array_equal(system.residue_types, expected)

    expected = np.array([LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"], LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"]], dtype=np.uint8)
    assert np.array_equal(system.loc_types, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.heavy_mask, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.hydrogen_mask, expected)

    expected = np.array([-44.471, 1.814, 9.610], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[0, HEAVY_ATOM_TYPE_ENCODER["SER"]["CA"]], expected)
    expected = np.array([-36.462, -15.953, -5.233], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[1, HEAVY_ATOM_TYPE_ENCODER["THR"]["CA"]], expected)
    expected = np.array([-41.975, -12.900, -1.374], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[2, HEAVY_ATOM_TYPE_ENCODER["PHE"]["CA"]], expected)


def test_system_from_pdb_file_with_ter_and_same_chain(tmp_path):
    # Create a temporary PDB file
    tmp_file = tmp_path / "test.pdb"
    tmp_file.write_text(
        "ATOM      1  N   SER A   1     -44.926   2.734  10.697  1.00  0.00           N  \n"
        "ATOM      2  CA  SER A   1     -44.471   1.814   9.610  1.00  0.00           C  \n"
        "ATOM      3  C   SER A   1     -45.511   1.713   8.489  1.00  0.00           C  \n"
        "ATOM      4  O   SER A   1     -46.707   1.747   8.768  1.00  0.00           O  \n"
        "ATOM      5  H   SER A   1     -45.804   2.417  11.094  1.00  0.00           H  \n"
        "ATOM      6  N   THR A   2     -36.145 -16.604  -6.518  1.00  0.00           N  \n"
        "ATOM      7  CA  THR A   2     -36.462 -15.953  -5.233  1.00  0.00           C  \n"
        "ATOM      8  C   THR A   2     -37.915 -15.465  -5.246  1.00  0.00           C  \n"
        "ATOM      9  O   THR A   2     -38.451 -15.183  -6.318  1.00  0.00           O  \n"
        "ATOM     10  H   THR A   2     -36.635 -16.243  -7.334  1.00  0.00           H  \n"
        "ATOM     11  N   PHE A   3     -40.964 -13.073  -2.428  1.00  0.00           N  \n"
        "ATOM     12  CA  PHE A   3     -41.975 -12.900  -1.374  1.00  0.00           C  \n"
        "ATOM     13  C   PHE A   3     -43.071 -11.881  -1.781  1.00  0.00           C  \n"
        "ATOM     14  O   PHE A   3     -42.740 -10.835  -2.391  1.00  0.00           O  \n"
        "ATOM     15  H   PHE A   3     -41.190 -12.602  -3.296  1.00  0.00           H  \n"
        "TER\n"
        "ATOM      1  N   ALA A   1     -44.926   2.734  10.697  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1     -44.471   1.814   9.610  1.00  0.00           C  \n"
        "ATOM      3  C   ALA A   1     -45.511   1.713   8.489  1.00  0.00           C  \n"
        "ATOM      4  O   ALA A   1     -46.707   1.747   8.768  1.00  0.00           O  \n"
        "ATOM      5  H   ALA A   1     -45.804   2.417  11.094  1.00  0.00           H  \n"
        "ATOM      6  N   PRO A   2     -36.145 -16.604  -6.518  1.00  0.00           N  \n"
        "ATOM      7  CA  PRO A   2     -36.462 -15.953  -5.233  1.00  0.00           C  \n"
        "ATOM      8  C   PRO A   2     -37.915 -15.465  -5.246  1.00  0.00           C  \n"
        "ATOM      9  O   PRO A   2     -38.451 -15.183  -6.318  1.00  0.00           O  \n"
        "ATOM     10  H   PRO A   2     -36.635 -16.243  -7.334  1.00  0.00           H  \n"
        "ATOM     11  HD3 PRO A   2     -46.213 -29.701 -29.444  1.00  0.00           H  \n"
        "ATOM     12  N   CYS A   3     -40.964 -13.073  -2.428  1.00  0.00           N  \n"
        "ATOM     13  CA  CYS A   3     -41.975 -12.900  -1.374  1.00  0.00           C  \n"
        "ATOM     14  C   CYS A   3     -43.071 -11.881  -1.781  1.00  0.00           C  \n"
        "ATOM     15  O   CYS A   3     -42.740 -10.835  -2.391  1.00  0.00           O  \n"
        "ATOM     16  H   CYS A   3     -41.190 -12.602  -3.296  1.00  0.00           H  \n"
        "TER\n"
    )

    # Initialize the system object
    system = system_from_pdb_path(tmp_file)

    # Assert correct initialization
    assert system.chain_ids.shape == (6,)
    assert system.residue_types.shape == (6,)
    assert system.loc_types.shape == (6,)
    assert system.heavy_coordinates.shape == (6, 15, 3)
    assert system.hydrogen_coordinates.shape == (6, 16, 3)
    assert system.heavy_mask.shape == (6,)
    assert system.hydrogen_mask.shape == (6,)

    expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32)
    assert np.array_equal(system.chain_ids, expected)

    expected = np.array([RESIDUE_TYPE_ENCODER["SER"], RESIDUE_TYPE_ENCODER["THR"], RESIDUE_TYPE_ENCODER["PHE"], RESIDUE_TYPE_ENCODER["ALA"], RESIDUE_TYPE_ENCODER["PRO"], RESIDUE_TYPE_ENCODER["CYS"]], dtype=np.uint8)
    assert np.array_equal(system.residue_types, expected)

    expected = np.array([LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"], LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"]], dtype=np.uint8)
    assert np.array_equal(system.loc_types, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.heavy_mask, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.hydrogen_mask, expected)

    expected = np.array([-44.471, 1.814, 9.610], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[0, HEAVY_ATOM_TYPE_ENCODER["SER"]["CA"]], expected)
    expected = np.array([-36.462, -15.953, -5.233], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[1, HEAVY_ATOM_TYPE_ENCODER["THR"]["CA"]], expected)
    expected = np.array([-41.975, -12.900, -1.374], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[2, HEAVY_ATOM_TYPE_ENCODER["PHE"]["CA"]], expected)


def test_system_from_cif_file(tmp_path):
    # Create a temporary PDB file
    tmp_file = tmp_path / "test.cif"
    tmp_file.write_text(
        "data_\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.pdbx_formal_charge\n"
        "ATOM 1  N N   . SER A 1 1 ? -44.926   2.734  10.697  1.00  0.00 ?\n"
        "ATOM 2  C CA  . SER A 1 1 ? -44.471   1.814   9.610  1.00  0.00 ?\n"
        "ATOM 3  C C   . SER A 1 1 ? -45.511   1.713   8.489  1.00  0.00 ?\n"
        "ATOM 4  O O   . SER A 1 1 ? -46.707   1.747   8.768  1.00  0.00 ?\n"
        "ATOM 5  H H   . SER A 1 1 ? -45.804   2.417  11.094  1.00  0.00 ?\n"
        "ATOM 6  N N   . THR A 1 2 ? -36.145 -16.604  -6.518  1.00  0.00 ?\n"
        "ATOM 7  C CA  . THR A 1 2 ? -36.462 -15.953  -5.233  1.00  0.00 ?\n"
        "ATOM 8  C C   . THR A 1 2 ? -37.915 -15.465  -5.246  1.00  0.00 ?\n"
        "ATOM 9  O O   . THR A 1 2 ? -38.451 -15.183  -6.318  1.00  0.00 ?\n"
        "ATOM 10 H H   . THR A 1 2 ? -36.635 -16.243  -7.334  1.00  0.00 ?\n"
        "ATOM 11 N N   . PHE A 1 3 ? -40.964 -13.073  -2.428  1.00  0.00 ?\n"
        "ATOM 12 C CA  . PHE A 1 3 ? -41.975 -12.900  -1.374  1.00  0.00 ?\n"
        "ATOM 13 C C   . PHE A 1 3 ? -43.071 -11.881  -1.781  1.00  0.00 ?\n"
        "ATOM 14 O O   . PHE A 1 3 ? -42.740 -10.835  -2.391  1.00  0.00 ?\n"
        "ATOM 15 H H   . PHE A 1 3 ? -41.190 -12.602  -3.296  1.00  0.00 ?\n"
        "ATOM 1  N N   . ALA B 2 1 ? -44.926   2.734  10.697  1.00  0.00 ?\n"
        "ATOM 2  C CA  . ALA B 2 1 ? -44.471   1.814   9.610  1.00  0.00 ?\n"
        "ATOM 3  C C   . ALA B 2 1 ? -45.511   1.713   8.489  1.00  0.00 ?\n"
        "ATOM 4  O O   . ALA B 2 1 ? -46.707   1.747   8.768  1.00  0.00 ?\n"
        "ATOM 5  H H   . ALA B 2 1 ? -45.804   2.417  11.094  1.00  0.00 ?\n"
        "ATOM 6  N N   . PRO B 2 2 ? -36.145 -16.604  -6.518  1.00  0.00 ?\n"
        "ATOM 7  C CA  . PRO B 2 2 ? -36.462 -15.953  -5.233  1.00  0.00 ?\n"
        "ATOM 8  C C   . PRO B 2 2 ? -37.915 -15.465  -5.246  1.00  0.00 ?\n"
        "ATOM 9  O O   . PRO B 2 2 ? -38.451 -15.183  -6.318  1.00  0.00 ?\n"
        "ATOM 10 H H   . PRO B 2 2 ? -36.635 -16.243  -7.334  1.00  0.00 ?\n"
        "ATOM 11 H HD3 . PRO B 2 2 ? -46.213 -29.701 -29.444  1.00  0.00 ?\n"
        "ATOM 12 N N   . CYS B 2 3 ? -40.964 -13.073  -2.428  1.00  0.00 ?\n"
        "ATOM 13 C CA  . CYS B 2 3 ? -41.975 -12.900  -1.374  1.00  0.00 ?\n"
        "ATOM 14 C C   . CYS B 2 3 ? -43.071 -11.881  -1.781  1.00  0.00 ?\n"
        "ATOM 15 O O   . CYS B 2 3 ? -42.740 -10.835  -2.391  1.00  0.00 ?\n"
        "ATOM 16 H H   . CYS B 2 3 ? -41.190 -12.602  -3.296  1.00  0.00 ?\n"
    )

    # Initialize the system object
    system = system_from_cif_path(tmp_file)

    # Assert correct initialization
    assert system.chain_ids.shape == (6,)
    assert system.residue_types.shape == (6,)
    assert system.loc_types.shape == (6,)
    assert system.heavy_coordinates.shape == (6, 15, 3)
    assert system.hydrogen_coordinates.shape == (6, 16, 3)
    assert system.heavy_mask.shape == (6,)
    assert system.hydrogen_mask.shape == (6,)

    expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32)
    assert np.array_equal(system.chain_ids, expected)

    expected = np.array([RESIDUE_TYPE_ENCODER["SER"], RESIDUE_TYPE_ENCODER["THR"], RESIDUE_TYPE_ENCODER["PHE"], RESIDUE_TYPE_ENCODER["ALA"], RESIDUE_TYPE_ENCODER["PRO"], RESIDUE_TYPE_ENCODER["CYS"]], dtype=np.uint8)
    assert np.array_equal(system.residue_types, expected)

    expected = np.array([LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"], LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"]], dtype=np.uint8)
    assert np.array_equal(system.loc_types, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.heavy_mask, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.hydrogen_mask, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.heavy_mask, expected)

    expected = np.array([True, True, True, True, True, True], dtype=np.bool)
    assert np.array_equal(system.hydrogen_mask, expected)

    expected = np.array([-44.471, 1.814, 9.610], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[0, HEAVY_ATOM_TYPE_ENCODER["SER"]["CA"]], expected)
    expected = np.array([-36.462, -15.953, -5.233], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[1, HEAVY_ATOM_TYPE_ENCODER["THR"]["CA"]], expected)
    expected = np.array([-41.975, -12.900, -1.374], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[2, HEAVY_ATOM_TYPE_ENCODER["PHE"]["CA"]], expected)


def test_system_from_pdb_partially_complete(tmp_path):
    # Create a temporary PDB file
    tmp_file = tmp_path / "test.pdb"
    tmp_file.write_text(
        "ATOM      1  N   GLY A   1      -1.278   1.972 -11.848\n"
        "ATOM      2  CA  GLY A   1      -1.859   0.751 -12.495\n"
        "ATOM      3  C   GLY A   1      -2.011  -0.290 -11.414\n"
        "ATOM      4  O   GLY A   1      -2.955  -0.209 -10.611\n"
        "ATOM      8 HA2  GLY A   1      -1.246   0.416 -13.210\n"
        "ATOM      9 HA3  GLY A   1      -2.751   0.965 -12.893\n"
        "ATOM     10  N   TRP A   2      -1.091  -1.212 -11.350\n"
        "ATOM     11  CA  TRP A   2      -1.168  -2.279 -10.380\n"
        "ATOM     12  C   TRP A   2      -0.710  -1.801  -8.991\n"
        "ATOM     13  O   TRP A   2      -0.886  -2.531  -8.014\n"
        "ATOM     14  CB  TRP A   2      -0.535  -3.589 -10.928\n"
        "ATOM     15  CG  TRP A   2      -1.286  -4.177 -12.112\n"
        "ATOM     16 CD1  TRP A   2      -2.440  -3.705 -12.675\n"
        "ATOM     17 NE1  TRP A   2      -2.871  -4.513 -13.704\n"
        "ATOM     18 CE2  TRP A   2      -1.992  -5.575 -13.836\n"
        "ATOM     19 CZ2  TRP A   2      -2.014  -6.660 -14.713\n"
        "ATOM     20 CH2  TRP A   2      -0.995  -7.553 -14.619\n"
        "ATOM     21 CZ3  TRP A   2       0.058  -7.408 -13.683\n"
        "ATOM     22 CE3  TRP A   2       0.073  -6.331 -12.793\n"
        "ATOM     23 CD2  TRP A   2      -0.951  -5.390 -12.846\n"
        "ATOM     24  H   TRP A   2      -0.318  -1.179 -11.983\n"
        "ATOM     25  HA  TRP A   2      -2.140  -2.488 -10.275\n"
        "ATOM     26 HB2  TRP A   2       0.402  -3.391 -11.215\n"
        "ATOM     27 HB3  TRP A   2      -0.524  -4.266 -10.193\n"
        "ATOM     28 HD1  TRP A   2      -2.912  -2.876 -12.376\n"
        "ATOM     29 HE1  TRP A   2      -3.685  -4.359 -14.263\n"
        "ATOM     30 HE3  TRP A   2       0.792  -6.204 -12.109\n"
        "ATOM     31 HZ2  TRP A   2      -2.732  -6.788 -15.397\n"
        "ATOM     32 HZ3  TRP A   2       0.794  -8.085 -13.646\n"
        "ATOM     33 HH2  TRP A   2      -1.005  -8.323 -15.256\n"
        "ATOM    294  N   ALA A   3       6.183  -0.108  14.517\n"
        "ATOM    295  CA  ALA A   3       6.751   0.968  15.310\n"
        "ATOM    296  C   ALA A   3       5.955   1.227  16.596\n"
        "ATOM    297  O   ALA A   3       6.550   1.949  17.453\n"
        "ATOM    299  CB  ALA A   3       6.833   2.250  14.479\n"
        "ATOM    300  H   ALA A   3       5.494   0.109  13.825\n"
        "ATOM    301  HA  ALA A   3       7.682   0.716  15.574\n"
        "ATOM    302 HB1  ALA A   3       7.242   2.982  15.024\n"
        "ATOM    303 HB2  ALA A   3       7.387   2.091  13.663\n"
        "ATOM    304 HB3  ALA A   3       5.902   2.503  14.215\n"
        "TER\n"
        "ATOM      1  N   GLY B   1      -1.278   1.972 -11.848\n"
        "ATOM      2  CA  GLY B   1      -1.859   0.751 -12.495\n"
        "ATOM      3  C   GLY B   1      -2.011  -0.290 -11.414\n"
        "ATOM      4  O   GLY B   1      -2.955  -0.209 -10.611\n"
        "ATOM      5  H1  GLY B   1      -1.891   2.306 -11.133\n"
        "ATOM      6  H2  GLY B   1      -0.392   1.742 -11.444\n"
        "ATOM      7  H3  GLY B   1      -1.155   2.685 -12.538\n"
        "ATOM      8 HA2  GLY B   1      -1.246   0.416 -13.210\n"
        "ATOM      9 HA3  GLY B   1      -2.751   0.965 -12.893\n"
        "ATOM     10  N   TRP B   2      -1.091  -1.212 -11.350\n"
        "ATOM     11  CA  TRP B   2      -1.168  -2.279 -10.380\n"
        "ATOM     12  C   TRP B   2      -0.710  -1.801  -8.991\n"
        "ATOM     13  O   TRP B   2      -0.886  -2.531  -8.014\n"
        "ATOM     14  CB  TRP B   2      -0.535  -3.589 -10.928\n"
        "ATOM     15  CG  TRP B   2      -1.286  -4.177 -12.112\n"
        "ATOM     16 CD1  TRP B   2      -2.440  -3.705 -12.675\n"
        "ATOM     18 CE2  TRP B   2      -1.992  -5.575 -13.836\n"
        "ATOM     19 CZ2  TRP B   2      -2.014  -6.660 -14.713\n"
        "ATOM     20 CH2  TRP B   2      -0.995  -7.553 -14.619\n"
        "ATOM     21 CZ3  TRP B   2       0.058  -7.408 -13.683\n"
        "ATOM     22 CE3  TRP B   2       0.073  -6.331 -12.793\n"
        "ATOM     23 CD2  TRP B   2      -0.951  -5.390 -12.846\n"
        "ATOM     24  H   TRP B   2      -0.318  -1.179 -11.983\n"
        "ATOM     25  HA  TRP B   2      -2.140  -2.488 -10.275\n"
        "ATOM     26 HB2  TRP B   2       0.402  -3.391 -11.215\n"
        "ATOM     27 HB3  TRP B   2      -0.524  -4.266 -10.193\n"
        "ATOM     28 HD1  TRP B   2      -2.912  -2.876 -12.376\n"
        "ATOM     30 HE3  TRP B   2       0.792  -6.204 -12.109\n"
        "ATOM     31 HZ2  TRP B   2      -2.732  -6.788 -15.397\n"
        "ATOM     32 HZ3  TRP B   2       0.794  -8.085 -13.646\n"
        "ATOM     33 HH2  TRP B   2      -1.005  -8.323 -15.256\n"
        "ATOM    294  N   ALA B   3       6.183  -0.108  14.517\n"
        "ATOM    295  CA  ALA B   3       6.751   0.968  15.310\n"
        "ATOM    296  C   ALA B   3       5.955   1.227  16.596\n"
        "ATOM    297  O   ALA B   3       6.550   1.949  17.453\n"
        "ATOM    298 OXT  ALA B   3       4.818   0.767  16.783\n"
        "ATOM    299  CB  ALA B   3       6.833   2.250  14.479\n"
        "ATOM    300  H   ALA B   3       5.494   0.109  13.825\n"
        "ATOM    301  HA  ALA B   3       7.682   0.716  15.574\n"
        "ATOM    302 HB1  ALA B   3       7.242   2.982  15.024\n"
        "ATOM    303 HB2  ALA B   3       7.387   2.091  13.663\n"
        "ATOM    304 HB3  ALA B   3       5.902   2.503  14.215\n"
        "TER\n"
    )

    # Initialize the system object
    system = system_from_pdb_path(tmp_file)

    # Assert correct initialization
    assert system.chain_ids.shape == (6,)
    assert system.residue_types.shape == (6,)
    assert system.loc_types.shape == (6,)
    assert system.heavy_coordinates.shape == (6, 15, 3)
    assert system.hydrogen_coordinates.shape == (6, 16, 3)
    assert system.heavy_mask.shape == (6,)
    assert system.hydrogen_mask.shape == (6,)

    expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32)
    assert np.array_equal(system.chain_ids, expected)

    expected = np.array([RESIDUE_TYPE_ENCODER["GLY"], RESIDUE_TYPE_ENCODER["TRP"], RESIDUE_TYPE_ENCODER["ALA"], RESIDUE_TYPE_ENCODER["GLY"], RESIDUE_TYPE_ENCODER["TRP"], RESIDUE_TYPE_ENCODER["ALA"]], dtype=np.uint8)
    assert np.array_equal(system.residue_types, expected)

    expected = np.array([LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"], LOC_TYPE_ENCODER["NTER"], LOC_TYPE_ENCODER["MIDDLE"], LOC_TYPE_ENCODER["CTER"]], dtype=np.uint8)
    assert np.array_equal(system.loc_types, expected)

    expected = np.array([False, False, True, False, True, False], dtype=np.bool)
    assert np.array_equal(system.heavy_mask, expected)

    expected = np.array([True, False, False, False, True, False], dtype=np.bool)
    assert np.array_equal(system.hydrogen_mask, expected)

    expected = np.array([-1.278, 1.972, -11.848], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[0, HEAVY_ATOM_TYPE_ENCODER["GLY"]["N"]], expected)
    expected = np.array([-0.710, -1.801, -8.991], dtype=np.float32)
    assert np.allclose(system.heavy_coordinates[1, HEAVY_ATOM_TYPE_ENCODER["TRP"]["C"]], expected)

    expected = np.array([-1.891, 2.306, -11.133], dtype=np.float32)
    assert np.allclose(system.hydrogen_coordinates[3, HYDROGEN_ATOM_TYPE_ENCODER["GLY"]["H1"]], expected)
    expected = np.array([7.242, 2.982, 15.024], dtype=np.float32)
    assert np.allclose(system.hydrogen_coordinates[5, HYDROGEN_ATOM_TYPE_ENCODER["ALA"]["HB1"]], expected)

def test_system_from_pdb_with_one_atom_residues(tmp_path):
    # Create a temporary PDB file
    tmp_file = tmp_path / "test.pdb"
    tmp_file.write_text(
        "ATOM      1 CA   ARG A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "TER\n"
        "ATOM      2 CA   TYR B   1       2.000   0.000   0.000  1.00  0.00           C  \n"
        "TER\n"
        "ATOM      3 CA   TYR C   1      -2.000   0.000   0.000  1.00  0.00           C  \n"
        "TER\n"
        "ATOM      4 CA   TYR D   1       0.000  -2.000   0.000  1.00  0.00           C  \n"
        "TER\n"
        "ATOM      5 CA   TYR E   1       0.000   0.000   2.000  1.00  0.00           C  \n"
        "TER\n"
        "ATOM      6 CA   TYR F   1       0.000   0.000  -2.000  1.00  0.00           C  \n"
        "TER\n"
        "END\n"
    )

    # Initialize the system object
    system = system_from_pdb_path(tmp_file)

    # Assert correct initialization
    assert system.chain_ids.shape == (6,)
    assert system.residue_types.shape == (6,)
    assert system.loc_types.shape == (6,)
    assert system.heavy_coordinates.shape == (6, 15, 3)
    assert system.hydrogen_coordinates.shape == (6, 16, 3)
    assert system.heavy_mask.shape == (6,)
    assert system.hydrogen_mask.shape == (6,)

    expected = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
    assert np.array_equal(system.chain_ids, expected)

    expected = np.array([RESIDUE_TYPE_ENCODER["ARG"], RESIDUE_TYPE_ENCODER["TYR"], RESIDUE_TYPE_ENCODER["TYR"], RESIDUE_TYPE_ENCODER["TYR"], RESIDUE_TYPE_ENCODER["TYR"], RESIDUE_TYPE_ENCODER["TYR"]], dtype=np.uint8)
    assert np.array_equal(system.residue_types, expected)

    expected = np.full((6,), LOC_TYPE_ENCODER["NTER"] | LOC_TYPE_ENCODER["CTER"], dtype=np.uint8)
    assert np.array_equal(system.loc_types, expected)

    expected = np.full((6,), True, dtype=np.bool)
    assert np.array_equal(system.heavy_mask, expected)

    expected = np.full((6,), True, dtype=np.bool)
    assert np.array_equal(system.hydrogen_mask, expected)

    assert np.isnan(system.heavy_coordinates[0, HEAVY_ATOM_TYPE_ENCODER["ARG"]["N"]]).all()
    assert np.isnan(system.heavy_coordinates[1, HEAVY_ATOM_TYPE_ENCODER["TYR"]["C"]]).all()
    assert np.isnan(system.hydrogen_coordinates[3, HYDROGEN_ATOM_TYPE_ENCODER["TYR"]["H1"]]).all()
    assert np.isnan(system.hydrogen_coordinates[5, HYDROGEN_ATOM_TYPE_ENCODER["TYR"]["HB2"]]).all()
