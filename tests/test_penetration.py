import pytest
import numpy as np
from cgback.parser import system_from_pdb_path
from cgback.penetration import find_residues_with_rings


@pytest.fixture
def system(tmp_path):
    # Create a temporary PDB file
    tmp_file = tmp_path / "test.pdb"
    tmp_file.write_text(
        "ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  SER A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      3  C   SER A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      4  O   SER A   1       0.000   0.000   0.000  1.00  0.00           O  \n"
        "ATOM      5  H   SER A   1       0.000   0.000   0.000  1.00  0.00           H  \n"
        "ATOM      6  N   THR A   2       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      7  CA  THR A   2       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      8  C   THR A   2       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      9  O   THR A   2       0.000   0.000   0.000  1.00  0.00           O  \n"
        "ATOM     10  H   THR A   2       0.000   0.000   0.000  1.00  0.00           H  \n"
        "ATOM     11  N   PHE A   3       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM     12  CA  PHE A   3       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM     13  C   PHE A   3       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM     14  O   PHE A   3       0.000   0.000   0.000  1.00  0.00           O  \n"
        "ATOM     15  H   PHE A   3       0.000   0.000   0.000  1.00  0.00           H  \n"
        "ATOM      1  N   ALA B   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      3  C   ALA B   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      4  O   ALA B   1       0.000   0.000   0.000  1.00  0.00           O  \n"
        "ATOM      5  H   ALA B   1       0.000   0.000   0.000  1.00  0.00           H  \n"
        "ATOM      6  N   PRO B   2       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      7  CA  PRO B   2       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      8  C   PRO B   2       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      9  O   PRO B   2       0.000   0.000   0.000  1.00  0.00           O  \n"
        "ATOM     10  H   PRO B   2       0.000   0.000   0.000  1.00  0.00           H  \n"
        "ATOM     11  HD3 PRO B   2       0.000   0.000   0.000  1.00  0.00           H  \n"
        "ATOM     12  N   CYS B   3       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM     13  CA  CYS B   3       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM     14  C   CYS B   3       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM     15  O   CYS B   3       0.000   0.000   0.000  1.00  0.00           O  \n"
        "ATOM     16  H   CYS B   3       0.000   0.000   0.000  1.00  0.00           H  \n"
    )

    # Initialize the system object
    system = system_from_pdb_path(tmp_file)

    return system


def test_find_residues_with_rings(system):
    # Find residues with rings
    indexes = find_residues_with_rings(system)

    assert np.array_equal(indexes, np.array([2, 4], dtype=np.uint64))
