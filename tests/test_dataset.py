import numpy as np

from cgback.dataset import process_mask, build_dataset


def test_process_mask():
    mask = "----"
    new_mask = np.array([False, False, False, False])
    assert np.array_equal(process_mask(mask), new_mask)

    mask = "++++"
    new_mask = np.array([True, True, True, True])
    assert np.array_equal(process_mask(mask), new_mask)

    mask = "-+++"
    new_mask = np.array([False, False, True, True])
    assert np.array_equal(process_mask(mask), new_mask)

    mask = "+-++"
    new_mask = np.array([False, False, False, True])
    assert np.array_equal(process_mask(mask), new_mask)

    mask = "++-+"
    new_mask = np.array([True, False, False, False])
    assert np.array_equal(process_mask(mask), new_mask)

    mask = "+++-"
    new_mask = np.array([True, True, False, False])
    assert np.array_equal(process_mask(mask), new_mask)


def test_build_dataset():
    np.random.seed(42)

    crd = [np.full((5, 15, 3), np.nan)]
    # L
    crd[0][0][0] = np.random.randn(3)  # N
    crd[0][0][1] = np.random.randn(3)  # CA
    crd[0][0][2] = np.random.randn(3)  # C
    crd[0][0][3] = np.random.randn(3)  # O
    crd[0][0][5] = np.random.randn(3)  # CB
    crd[0][0][6] = np.random.randn(3)  # CG
    crd[0][0][7] = np.random.randn(3)  # CD1
    crd[0][0][8] = np.random.randn(3)  # CD2
    # I
    crd[0][1][0] = np.random.randn(3)  # N
    crd[0][1][1] = np.random.randn(3)  # CA
    crd[0][1][2] = np.random.randn(3)  # C
    crd[0][1][3] = np.random.randn(3)  # O
    crd[0][1][5] = np.random.randn(3)  # CB
    crd[0][1][6] = np.random.randn(3)  # CG1
    crd[0][1][7] = np.random.randn(3)  # CD1
    crd[0][1][8] = np.random.randn(3)  # CG2
    # P
    crd[0][2][0] = np.random.randn(3)  # N
    crd[0][2][1] = np.random.randn(3)  # CA
    crd[0][2][2] = np.random.randn(3)  # C
    crd[0][2][3] = np.random.randn(3)  # O
    crd[0][2][5] = np.random.randn(3)  # CB
    crd[0][2][6] = np.random.randn(3)  # CG
    crd[0][2][7] = np.random.randn(3)  # CD
    # I
    crd[0][3][0] = np.random.randn(3)  # N
    crd[0][3][1] = np.random.randn(3)  # CA
    crd[0][3][2] = np.random.randn(3)  # C
    crd[0][3][3] = np.random.randn(3)  # O
    crd[0][3][5] = np.random.randn(3)  # CB
    crd[0][3][6] = np.random.randn(3)  # CG1
    crd[0][3][7] = np.random.randn(3)  # CD1
    crd[0][3][8] = np.random.randn(3)  # CG2
    # D
    crd[0][4][0] = np.random.randn(3)  # N
    crd[0][4][1] = np.random.randn(3)  # CA
    crd[0][4][2] = np.random.randn(3)  # C
    crd[0][4][3] = np.random.randn(3)  # O
    crd[0][4][5] = np.random.randn(3)  # CB
    crd[0][4][6] = np.random.randn(3)  # CG
    crd[0][4][7] = np.random.randn(3)  # OD1
    crd[0][4][8] = np.random.randn(3)  # OD2
    seq = ["LIPID"]
    msk = ["+++++"]
    cut = 10.0
    dataset = build_dataset(crd, seq, msk, cut)
    assert np.allclose(dataset["w"][0], crd[0][0][1])
    assert np.allclose(dataset["w"][5], crd[0][1][1])
    assert np.allclose(dataset["w"][10], crd[0][2][1])
    assert np.allclose(dataset["w"][15], crd[0][3][1])
    assert np.allclose(dataset["w"][20], crd[0][4][1])
    assert np.array_equal(dataset["h_offset"], np.array([0, 7, 14, 20, 27, 34]))
    assert np.array_equal(dataset["u_offset"], np.array([0, 5, 10, 15, 20, 25]))
