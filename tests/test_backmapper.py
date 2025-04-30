import math
from pytest import fixture
from pathlib import Path
from types import SimpleNamespace
from cgback.backmapper import Backmapper

@fixture
def args():
    return SimpleNamespace(
        INPUT=None,
        output="out.pdb",
        model="M",
        num_timesteps=None,
        device="cpu",
        batch=1024,
        seed=42,
        keep=False,
        energy_minimization=False,
        ignore_existing=False,
        verbose=False,
        help=False,
        diffuser="DDPM",
        fix_structure_model="M",
        fix_structure_max_iterations=1000,
        energy_minimization_cutoff=30.0,
        energy_minimization_tolerance=10.0,
        energy_minimization_max_iterations=0,
        energy_minimization_log_interval=50,
        energy_minimization_ignore_existing=False,
        skip_sampling=False,
        skip_add_hydrogen=False,
        skip_fix_structure=False,
        skip_fix_structure_clashes=False,
        custom_model_checkpoint_path=None,
        custom_model_num_timesteps=None,
        custom_model_num_layers=None,
        custom_model_dim_hidden=None,
        custom_model_cutoff=None,
        debug=False,
    )

def test_cgback_app_initialization(args):
    args.INPUT = "foo"
    Backmapper(args)

def test_cgback_app_run(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_2lcn.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_2lcn.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_with_model_s(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_2lcn.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.model = "S"

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_2lcn_model_s.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_with_model_m(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_2lcn.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.model = "M"

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_2lcn_model_m.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_with_model_l(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_2lcn.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.model = "L"

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_2lcn_model_l.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_with_cif_format(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_1yiw.cif"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.cif"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.model = "S"

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_1yiw.cif"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:40] == ref[:40]
            assert math.isclose(float(out[40:50].strip()), float(ref[40:50].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[51:61].strip()), float(ref[51:61].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[62:72].strip()), float(ref[62:72].strip()), abs_tol=1e-3)
            assert out[72:] == ref[72:]
        else:
            assert out == ref

def test_cgback_app_run_with_ring_penetration(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_1a0s.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.skip_fix_structure_clashes = True

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_1a0s.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_with_different_seeds(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_2lcn.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.model = "S"

    args.seed = 0
    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_2lcn_seed_0.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    assert out_lines == ref_lines

    args.seed = 1
    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_2lcn_seed_1.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref


def test_cgback_app_run_skip(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "aa_2lcn.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.skip_sampling = True
    args.skip_add_hydrogen = True
    args.skip_fix_structure = True

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_2lcn.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_identity(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "aa_2lcn.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_2lcn.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_ignore(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "aa_ignore.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.ignore_existing = True

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_ignore.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_pdb_partial(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_partial.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_partial.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_fix_aa_penetration(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "aa_1a0s_with_penetration.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file
    args.skip_fix_structure_clashes = True

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_1a0s_without_penetration.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref

def test_cgback_app_run_histidine_variants(tmp_path, args):
    inp_file = Path(__file__).parent / "data" / "cg_his.pdb"
    inp_file = str(inp_file)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_file = out_dir / "out.pdb"
    out_file = str(out_file)

    args.INPUT = inp_file
    args.output = out_file

    app = Backmapper(args)
    app.run()

    out_file = Path(out_file)
    ref_file = Path(__file__).parent / "data" / "aa_his.pdb"

    out_lines = out_file.read_text(encoding="utf-8").splitlines()
    ref_lines = ref_file.read_text(encoding="utf-8").splitlines()
    for out, ref in zip(out_lines, ref_lines):
        if out.startswith("ATOM"):
            assert out[:30] == ref[:30]
            assert math.isclose(float(out[30:38].strip()), float(ref[30:38].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[38:46].strip()), float(ref[38:46].strip()), abs_tol=1e-3)
            assert math.isclose(float(out[46:54].strip()), float(ref[46:54].strip()), abs_tol=1e-3)
            assert out[54:] == ref[54:]
        else:
            assert out == ref