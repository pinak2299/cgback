import math
import numpy as np
from numpy.typing import NDArray
from cgback.system import System
from cgback.parser import HEAVY_ATOM_TYPE_ENCODER, HYDROGEN_ATOM_TYPE_ENCODER, RESIDUE_TYPE_ENCODER, RESIDUE_TYPE_DECODER, LOC_TYPE_ENCODER

COS5475 = math.cos(math.pi * 54.75 / 180.0)
SIN5475 = math.sin(math.pi * 54.75 / 180.0)


def calculate_u12(r1: NDArray, r2: NDArray) -> NDArray:
    r12 = r2 - r1
    u12 = r12 / np.linalg.norm(r12)

    return u12


def calculate_cx3h1(r1: NDArray, r2: NDArray, r3: NDArray, r4: NDArray, d: float = 1.0) -> NDArray:
    r12 = r2 - r1
    r13 = r3 - r1

    u12 = r12 / np.linalg.norm(r12)
    u13 = r13 / np.linalg.norm(r13)

    a1 = np.cross(u12, u13)
    u1 = a1 / np.linalg.norm(a1)
    rc = (r1 + r2 + r3) / 3.0
    rc4 = r4 - rc
    if np.dot(rc4, u1) < 0.0:
        u1 = -u1

    h1 = r4 + d * u1

    return h1


def calculate_cx2h2(r1: NDArray, r2: NDArray, r3: NDArray, d: float = 1.0) -> tuple[NDArray, NDArray]:
    r12 = r2 - r1
    r13 = r3 - r1

    u12 = r12 / np.linalg.norm(r12)
    u13 = r13 / np.linalg.norm(r13)

    a1 = -(u12 + u13)
    u1 = a1 / np.linalg.norm(a1)
    a2 = np.cross(u12, u13)
    u2 = a2 / np.linalg.norm(a2)

    h1 = r1 + d * (COS5475 * u1 + SIN5475 * u2)
    h2 = r1 + d * (COS5475 * u1 - SIN5475 * u2)

    return h1, h2


def calculate_cx2h1(r1: NDArray, r2: NDArray, r3: NDArray, d: float = 1.0) -> NDArray:
    r12 = r2 - r1
    r13 = r3 - r1

    u12 = r12 / np.linalg.norm(r12)
    u13 = r13 / np.linalg.norm(r13)

    a1 = -(u12 + u13)
    u1 = a1 / np.linalg.norm(a1)

    h1 = r1 + d * u1

    return h1


def update_hydrogen_coordinates(system: System) -> None:
    # Iterate over residues
    for idx, (residue_type, loc_type, heavy_coordinates, hydrogen_coordinates, heavy_mask, hydrogen_mask) in enumerate(zip(system.residue_types, system.loc_types, system.heavy_coordinates, system.hydrogen_coordinates, system.heavy_mask, system.hydrogen_mask)):
        if not (heavy_mask or hydrogen_mask): continue
        residue_name = RESIDUE_TYPE_DECODER[residue_type]
        n = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["N"]]
        ca = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CA"]]
        c = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["C"]]

        # Histidine flags
        if residue_name == "HIS":
            his_hd1_flag = not np.isnan(hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD1"]]).all()
            his_he2_flag = not np.isnan(hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE2"]]).all()
            if not his_hd1_flag and not his_he2_flag: his_hd1_flag = True
        else:
            his_hd1_flag = False
            his_he2_flag = False

        # Initialize positions
        hydrogen_coordinates.fill(np.nan)

        # Add HA
        ha = None
        cb = None
        if residue_type != RESIDUE_TYPE_ENCODER["GLY"]:
            cb = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CB"]]
            ha = calculate_cx3h1(cb, c, n, ca)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HA"]] = ha

        if residue_name == "ALA":
            # Add HB3
            hb3 = cb + ca - ha
            # Add HB1 and HB2
            hb1, hb2 = calculate_cx2h2(cb, hb3, ca)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB1"]] = hb1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3

        elif residue_name == "ARG":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD"]]
            ne = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["NE"]]
            cz = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CZ"]]
            nh1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["NH1"]]
            nh2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["NH2"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HG2 and HG3
            hg2, hg3 = calculate_cx2h2(cg, cb, cd)
            # Add HD2 and HD3
            hd2, hd3 = calculate_cx2h2(cd, cg, ne)
            # Add HE
            he = calculate_cx2h1(ne, cz, cd)
            # Add HH11, HH12, HH21, and HH22
            hh11 = nh1 + calculate_u12(nh2, cz)
            hh12 = nh1 + calculate_u12(ne, cz)
            hh21 = nh2 + calculate_u12(nh1, cz)
            hh22 = nh2 + calculate_u12(ne, cz)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG2"]] = hg2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG3"]] = hg3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD2"]] = hd2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD3"]] = hd3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE"]] = he
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HH11"]] = hh11
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HH12"]] = hh12
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HH21"]] = hh21
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HH22"]] = hh22

        elif residue_name == "ASN":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            od1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["OD1"]]
            nd2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["ND2"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HD21 and HD22
            hd21 = nd2 + calculate_u12(cb, cg)
            hd22 = nd2 + calculate_u12(od1, cg)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD21"]] = hd21
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD22"]] = hd22

        elif residue_name == "ASP":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3

        elif residue_name == "CYS":
            sg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["SG"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, sg, ca)
            # Add HG
            hg = sg + calculate_u12(ca, cb)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG"]] = hg

        elif residue_name == "GLN":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD"]]
            oe1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["OE1"]]
            oe1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["OE1"]]
            ne2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["NE2"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HG2 and HG3
            hg2, hg3 = calculate_cx2h2(cg, cb, cd)
            # Add HE21 and HE22
            he21 = ne2 + calculate_u12(oe1, cd)
            he22 = ne2 + calculate_u12(cg, cd)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG2"]] = hg2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG3"]] = hg3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE21"]] = he21
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE22"]] = he22

        elif residue_name == "GLU":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HG2 and HG3
            hg2, hg3 = calculate_cx2h2(cg, cb, cd)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG2"]] = hg2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG3"]] = hg3

        elif residue_name == "GLY":
            # Add HA2 and HA3
            ha2, ha3 = calculate_cx2h2(ca, n, c)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HA2"]] = ha2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HA3"]] = ha3

        elif residue_name == "HIS":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            nd1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["ND1"]]
            ce1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE1"]]
            ne2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["NE2"]]
            cd2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD2"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HD1
            hd1 = calculate_cx2h1(nd1, cg, ce1)
            # Add HE1
            he1 = calculate_cx2h1(ce1, nd1, ne2)
            # Add HD2
            hd2 = calculate_cx2h1(cd2, cg, ne2)
            # Add HE2
            he2 = calculate_cx2h1(ne2, cd2, ce1)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE1"]] = he1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD2"]] = hd2
            if his_hd1_flag: hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD1"]] = hd1
            if his_he2_flag: hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE2"]] = he2

        elif residue_name == "ILE":
            cg1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG1"]]
            cd1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD1"]]
            cg2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG2"]]
            # Add HB
            hb = calculate_cx3h1(ca, cg2, cg1, cb)
            # Add HG12 and HG13
            hg12, hg13 = calculate_cx2h2(cg1, cd1, cb)
            # Add HG21
            hg21 = cg2 + calculate_u12(hb, cb)
            # Add HG22 and HG23
            hg22, hg23 = calculate_cx2h2(cg2, cb, hg21)
            # Add HD11
            hd11 = cd1 + calculate_u12(cb, cg1)
            # Add HD12 and HD13
            hd12, hd13 = calculate_cx2h2(cd1, cg1, hd11)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB"]] = hb
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG12"]] = hg12
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG13"]] = hg13
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG21"]] = hg21
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG22"]] = hg22
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG23"]] = hg23
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD11"]] = hd11
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD12"]] = hd12
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD13"]] = hd13

        elif residue_name == "LEU":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD1"]]
            cd2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD2"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HG
            hg = calculate_cx3h1(cb, cd1, cd2, cg)
            # Add HD11
            hd11 = cd1 + calculate_u12(hg, cg)
            # Add HD12 and HD13
            hd12, hd13 = calculate_cx2h2(cd1, hd11, cg)
            # Add HD21
            hd21 = cd2 + calculate_u12(hg, cg)
            # Add HD22 and HD23
            hd22, hd23 = calculate_cx2h2(cd2, hd21, cg)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG"]] = hg
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD11"]] = hd11
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD12"]] = hd12
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD13"]] = hd13
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD21"]] = hd21
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD22"]] = hd22
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD23"]] = hd23

        elif residue_name == "LYS":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD"]]
            ce = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE"]]
            nz = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["NZ"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HG2 and HG3
            hg2, hg3 = calculate_cx2h2(cg, cb, cd)
            # Add HD2 and HD3
            hd2, hd3 = calculate_cx2h2(cd, cg, ce)
            # Add HE2 and HE3
            he2, he3 = calculate_cx2h2(ce, cd, nz)
            # Add HZ1
            hz1 = nz + calculate_u12(cd, ce)
            # Add HZ2 and HZ3
            hz2, hz3 = calculate_cx2h2(nz, hz1, ce)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG2"]] = hg2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG3"]] = hg3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD2"]] = hd2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD3"]] = hd3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE2"]] = he2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE3"]] = he3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HZ1"]] = hz1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HZ2"]] = hz2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HZ3"]] = hz3

        elif residue_name == "MET":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            sd = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["SD"]]
            ce = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HG2 and HG3
            hg2, hg3 = calculate_cx2h2(cg, cb, sd)
            # Add HE1
            he1 = ce + calculate_u12(cg, sd)
            # Add HE2 and HE3
            he2, he3 = calculate_cx2h2(ce, he1, sd)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG2"]] = hg2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG3"]] = hg3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE1"]] = he1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE2"]] = he2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE3"]] = he3

        elif residue_name == "PHE":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD1"]]
            ce1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE1"]]
            cz = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CZ"]]
            ce2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE2"]]
            cd2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD2"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HD1
            hd1 = cd1 + calculate_u12(cd2, cg)
            # Add HD2
            hd2 = cd2 + calculate_u12(cd1, cg)
            # Add HE1
            he1 = ce1 + calculate_u12(cg, cd1)
            # Add HE2
            he2 = ce2 + calculate_u12(cg, cd2)
            # Add HZ
            hz = cz + calculate_u12(cb, cg)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD1"]] = hd1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD2"]] = hd2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE1"]] = he1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE2"]] = he2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HZ"]] = hz

        elif residue_name == "PRO":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HG2 and HG3
            hg2, hg3 = calculate_cx2h2(cg, cb, cd)
            # Add HD2 and HD3
            hd2, hd3 = calculate_cx2h2(cd, cg, n)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG2"]] = hg2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG3"]] = hg3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD2"]] = hd2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD3"]] = hd3

        elif residue_name == "SER":
            og = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["OG"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, og, ca)
            # Add HG
            hg = og + calculate_u12(ca, cb)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG"]] = hg

        elif residue_name == "THR":
            og1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["OG1"]]
            cg2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG2"]]
            # Add HB
            hb = calculate_cx3h1(ca, cg2, og1, cb)
            # Add HG1
            hg1 = og1 + calculate_u12(ca, cb)
            # Add HG21
            hg21 = cg2 + calculate_u12(ca, cb)
            # Add HG22 and HG23
            hg22, hg23 = calculate_cx2h2(cg2, hg21, cb)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB"]] = hb
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG1"]] = hg1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG21"]] = hg21
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG22"]] = hg22
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG23"]] = hg23

        elif residue_name == "TRP":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD1"]]
            ne1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["NE1"]]
            ce2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE2"]]
            cz2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CZ2"]]
            ch2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CH2"]]
            cz3 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CZ3"]]
            ce3 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE3"]]
            cd2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD2"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HD1
            hd1 = calculate_cx2h1(cd1, ne1, cg)
            # Add HE1
            he1 = calculate_cx2h1(ne1, ce2, cd1)
            # Add HE3
            he3 = ce3 + calculate_u12(ce2, cd2)
            # Add HZ2
            hz2 = cz2 + calculate_u12(cd2, ce2)
            # Add HZ3
            hz3 = cz3 + calculate_u12(cd2, ce3)
            # Add HH2
            hh2 = ch2 + calculate_u12(ce3, cz3)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD1"]] = hd1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE1"]] = he1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE3"]] = he3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HZ2"]] = hz2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HZ3"]] = hz3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HH2"]] = hh2

        elif residue_name == "TYR":
            cg = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG"]]
            cd1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD1"]]
            ce1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE1"]]
            oh = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["OH"]]
            ce2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CE2"]]
            cd2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD2"]]
            # Add HB2 and HB3
            hb2, hb3 = calculate_cx2h2(cb, cg, ca)
            # Add HD1
            hd1 = cd1 + calculate_u12(cd2, cg)
            # Add HD2
            hd2 = cd2 + calculate_u12(cd1, cg)
            # Add HE1
            he1 = ce1 + calculate_u12(cg, cd1)
            # Add HE2
            he2 = ce2 + calculate_u12(cg, cd2)
            # Add HH
            hh = oh + calculate_u12(cg, cd1)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB2"]] = hb2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB3"]] = hb3
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD1"]] = hd1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HD2"]] = hd2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE1"]] = he1
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HE2"]] = he2
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HH"]] = hh

        elif residue_name == "VAL":
            cg1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG1"]]
            cg2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG2"]]
            # Add HB
            hb = calculate_cx3h1(ca, cg1, cg2, cb)
            # Add HG11
            hg11 = cg1 + calculate_u12(hb, cb)
            # Add HG12 and HG13
            hg12, hg13 = calculate_cx2h2(cg1, hg11, cb)
            # Add HG21
            hg21 = cg2 + calculate_u12(hb, cb)
            # Add HG22 and HG23
            hg22, hg23 = calculate_cx2h2(cg2, cb, hg21)

            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HB"]] = hb
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG11"]] = hg11
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG12"]] = hg12
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG13"]] = hg13
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG21"]] = hg21
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG22"]] = hg22
            hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HG23"]] = hg23

        # Add H to N
        if loc_type & LOC_TYPE_ENCODER["NTER"] != 0:
            if residue_name == "PRO":
                cd = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CD"]]
                h1, h2 = calculate_cx2h2(n, ca, cd)

                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H1"]] = h1
                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H2"]] = h2

            elif residue_name == "GLY":
                ha2 = hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["HA2"]]
                h1 = n + ca - ha2
                h2, h3 = calculate_cx2h2(n, h1, ca)

                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H1"]] = h1
                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H2"]] = h2
                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H3"]] = h3

            else:
                h1 = n + ca - ha
                h2, h3 = calculate_cx2h2(n, h1, ca)

                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H1"]] = h1
                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H2"]] = h2
                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H3"]] = h3
        else:
            if residue_name != "PRO":
                residue_type_prev = system.residue_types[idx - 1]
                residue_name_prev = RESIDUE_TYPE_DECODER[residue_type_prev]
                c_prev = system.heavy_coordinates[idx - 1, HEAVY_ATOM_TYPE_ENCODER[residue_name_prev]["C"]]
                h = calculate_cx2h1(n, ca, c_prev)

                hydrogen_coordinates[HYDROGEN_ATOM_TYPE_ENCODER[residue_name]["H"]] = h

        # Add OXT
        if loc_type & LOC_TYPE_ENCODER["CTER"] != 0:
            o = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["O"]]
            oxt = calculate_cx2h1(c, o, ca, 1.24)

            heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["OXT"]] = oxt


def calculate_num_chiral_centers(system: System) -> int:
    num_chiral_centers = 0
    for idx, (residue_type, heavy_coordinates, hydrogen_coordinates) in enumerate(zip(system.residue_types, system.heavy_coordinates, system.hydrogen_coordinates)):
        # GLY does not contain a chirality center
        residue_name = RESIDUE_TYPE_DECODER[residue_type]
        if residue_name == "GLY": continue
        # All residues contain a chirality center at the Ca(L)
        num_chiral_centers += 1
        # ILE contains a chirality center at Cb(S)
        if residue_name == "ILE": num_chiral_centers += 1
        # THR contains a chirality center at Cb(R)
        elif residue_name == "THR": num_chiral_centers += 1

    return num_chiral_centers


def find_incorrect_chiral_centers(system: System) -> tuple[list[tuple[int,str]],int]:
    incorrect_chiral_centers = []
    num_chiral_centers_with_wrong_chirality = 0
    for idx, (residue_type, heavy_coordinates, hydrogen_coordinates) in enumerate(zip(system.residue_types, system.heavy_coordinates, system.hydrogen_coordinates)):
        # GLY does not contain a chirality center
        residue_name = RESIDUE_TYPE_DECODER[residue_type]
        if residue_name == "GLY": continue

        # All residues contain a chirality center at the Ca(L)
        n = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["N"]]
        ca = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CA"]]
        c = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["C"]]
        cb = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CB"]]
        judge = np.dot(cb-ca, np.cross(n-ca, c-ca))
        if judge < 0.0:
            num_chiral_centers_with_wrong_chirality += 1
            incorrect_chiral_centers.append((idx, "CA"))

        # ILE contains a chirality center at Cb(S)
        if residue_name == "ILE":
            cg1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG1"]]
            cg2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG2"]]
            judge = np.dot(cg2-cb, np.cross(ca-cb, cg1-cb))
            if judge < 0.0:
                num_chiral_centers_with_wrong_chirality += 1
                incorrect_chiral_centers.append((idx, "CB"))

        # THR contains a chirality center at Cb(R)
        elif residue_name == "THR":
            og1 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["OG1"]]
            cg2 = heavy_coordinates[HEAVY_ATOM_TYPE_ENCODER[residue_name]["CG2"]]
            judge = np.dot(cg2-cb, np.cross(og1-cb, ca-cb))
            if judge >= 0.0:
                num_chiral_centers_with_wrong_chirality += 1
                incorrect_chiral_centers.append((idx, "CB"))

    return incorrect_chiral_centers, num_chiral_centers_with_wrong_chirality
