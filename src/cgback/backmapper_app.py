import argparse
import importlib.metadata as metadata
from cgback.backmapper import Backmapper


def main() -> None:
    class CustomHelpFormatter(argparse.HelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=100, width=130)

    parser = argparse.ArgumentParser(description="CGBack is a tool for backmapping coarse-grained structures into all-atom models", add_help=False, formatter_class=CustomHelpFormatter)
    parser.add_argument("INPUT", metavar="FILE", type=str, help="Input PDB or CIF file")
    parser.add_argument("-o", "--output", metavar="FILE", type=str, default="out.pdb", help="Output PDB file (default: out.pdb)")
    parser.add_argument("-m", "--model", metavar="MODEL", type=str, choices=["S", "M", "L", "C"], default="M", help="Model size (default: M, choices=['S', 'M', 'L'])")
    parser.add_argument("-n", "--num-timesteps", metavar="STEPS", type=int, help="Number of diffusion time steps")
    parser.add_argument("-d", "--device", metavar="DEVICE", type=str, choices=["cpu", "cuda", "mps"], default="cpu", help="Device to use (default: cpu, choices=['cpu', 'cuda', 'mps'])")
    parser.add_argument("-b", "--batch", metavar="BATCH", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("-s", "--seed", metavar="SEED", type=int, default=42, help="Seed for RNG (default: 42)")
    parser.add_argument("-k", "--keep", action="store_true", help="Keep intermediate files")
    parser.add_argument("-e", "--energy-minimization", action="store_true", help="Perform energy minimization (requires OpenMM support)")
    parser.add_argument("-i", "--ignore-existing", action="store_true", help="Ignore existing backmapped residues in the structure")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Display this help message and exit")
    parser.add_argument("--diffuser", metavar="DIFFUSER", type=str, choices=["DDPM", "DDIM"], default="DDPM", help="Diffuser type (default: DDPM, choices=['DDPM', 'DDIM'])")
    parser.add_argument("--fix-structure-model", metavar="MODEL", type=str, choices=["S", "M", "L"], default="M", help="Model size for fixing structure artifacts (default: M, choices=['S', 'M', 'L'])")
    parser.add_argument("--fix-structure-max-iterations", metavar="NUMBER", type=int, default=1000, help="Max number of iterations for fixing structure artifacts (default: 1000)")
    parser.add_argument("--energy-minimization-cutoff", metavar="NUMBER", type=float, default=30.0, help="Cutoff distance in angstroms (default: 30.0)")
    parser.add_argument("--energy-minimization-tolerance", metavar="NUMBER", type=float, default=10.0, help="Tolerance in kJ/mol for OpenMM energy minimization (default: 10.0)")
    parser.add_argument("--energy-minimization-max-iterations", metavar="NUMBER", type=int, default=0, help="Max number of iterations for OpenMM energy minimization (default: 0)")
    parser.add_argument("--energy-minimization-log-interval", metavar="NUMBER", type=int, default=50, help="Frequency for logging energy minimization progress (default: 50)")
    parser.add_argument("--energy-minimization-ignore-existing", action="store_true", help="Avoid fixing positions of existing residues during energy minimization")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip sampling for regenerating heavy atom positions")
    parser.add_argument("--skip-add-hydrogen", action="store_true", help="Skip adding hydrogen atoms")
    parser.add_argument("--skip-fix-structure", action="store_true", help="Skip fixing structure artifacts")
    parser.add_argument("--skip-fix-structure-clashes", action="store_true", help="Skip fixing clashes during refinement")
    parser.add_argument('--version', action='version', version=metadata.version(__package__), help="Show version and exit")
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--custom-model-checkpoint-path", metavar="FILE", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--custom-model-num-timesteps", metavar="NUMBER", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--custom-model-num-layers", metavar="NUMBER", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--custom-model-dim-hidden", metavar="NUMBER", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--custom-model-cutoff", metavar="NUMBER", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()

    app = Backmapper(args)
    app.run()


if __name__ == "__main__":
    main()
