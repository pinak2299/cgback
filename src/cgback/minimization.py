import sys
import pathlib
import tempfile
import numpy as np
from logging import Logger
from cgback.system import System
from cgback.parser import write_cif_from_system, HEAVY_ATOM_TYPE_ENCODER, HYDROGEN_ATOM_TYPE_ENCODER, LOC_TYPE_ENCODER
from openmm import OpenMMException
from openmm.app import ForceField, CutoffNonPeriodic, PDBxFile
from openmm.openmm import Context, LocalEnergyMinimizer, MinimizationReporter, Platform, VerletIntegrator
from openmm.unit import angstroms, dalton, kilojoules_per_mole, nanometers, femtoseconds


class CustomMinimizationReporter(MinimizationReporter):
    def __init__(self, epsilon: float, interval=50, logger: Logger | None = None):
        super().__init__()
        self.interval = interval
        self.epsilon = epsilon
        self.logger = logger

    def report(self, iteration, x, grad, args=None):
        e_kj = args["system energy"]
        if iteration % self.interval == 0 and iteration > 0 and self.logger is not None:
            gnorm = np.sqrt(np.mean(np.square(grad)))
            xnorm = np.sqrt(np.mean(np.square(x)))
            xnorm = 1.0 if xnorm < 1.0 else xnorm
            judge = gnorm / xnorm
            self.logger.info(f"Iteration: {iteration:6d} | Energy: {e_kj:12.3f} kJ/mol | Judge: {judge:9.3f}")
        return False


def energy_minimization(cgback_system: System, cutoff: float = 30.0, iterations: int = 0, tolerance: float = 10.0, log_interval: int = 50, ignore_existing: bool = False, logger: Logger | None = None) -> None:
    with tempfile.NamedTemporaryFile() as tmp:
        write_cif_from_system(pathlib.Path(tmp.name), cgback_system)
        pdb = PDBxFile(tmp.name)
    force_field = ForceField("amber99sbildn.xml", "amber99_obc.xml")
    openmm_system = force_field.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=cutoff*angstroms, constraints=None)
    if logger is not None: logger.info(f"Performing energy minimization with force fields 'AMBER99SB-ILDN' and 'AMBER99 OBC'")
    if logger is not None: logger.info(f"Performing energy minimization with a non-bonded cutoff of {cutoff:.1f} Å")
    if logger is not None: logger.info(f"Performing energy minimization without PBC")

    if not ignore_existing:
        for atom in pdb.topology.atoms():
            if atom.name == "CA":
                openmm_system.setParticleMass(atom.index, 0 * dalton)

        for residue, system_heavy_mask, system_hydrogen_mask in zip(pdb.topology.residues(), cgback_system.heavy_mask, cgback_system.hydrogen_mask):
            if system_heavy_mask: continue
            for atom in residue.atoms():
                if atom.name.startswith("H") and system_hydrogen_mask: continue
                openmm_system.setParticleMass(atom.index, 0 * dalton)

    platform_names = []
    for i in range(Platform.getNumPlatforms()):
        platform_names.append(Platform.getPlatform(i).getName())
    platform_name = "CUDA" if "CUDA" in platform_names else "OpenCL" if "OpenCL" in platform_names else "CPU"
    platform = Platform.getPlatformByName(platform_name)
    if logger is not None: logger.info(f"Performing energy minimization with platform '{platform_name}'")

    try:
        integrator = VerletIntegrator(1.0 * femtoseconds)
        context = Context(openmm_system, integrator, platform)
        context.setPositions(pdb.positions)
        if logger is not None: logger.info(f"Performing energy minimization with {iterations} max iterations and a tolerance of {tolerance} kJ/mol")
    except OpenMMException as e:
        logger.error(f"Failed to initialize OpenMM context ({str(e)})")
        sys.exit(0)

    positions = context.getState(getPositions=True).getPositions(asNumpy=True) / nanometers
    mean = np.mean(np.sum(np.square(positions), axis=1))
    norm = 1.0 if mean < 1.0 else np.sqrt(mean)
    epsilon = tolerance / norm
    epsilon = float(epsilon)
    if logger is not None: logger.info(f"Performing energy minimization with epsilon {epsilon:.3f}")

    tolerance = tolerance * kilojoules_per_mole / nanometers
    max_iterations = iterations
    LocalEnergyMinimizer.minimize(context, tolerance, max_iterations, reporter=CustomMinimizationReporter(epsilon, log_interval, logger))

    state = context.getState(getPositions=True)
    minimized_positions = state.getPositions(asNumpy=True) / angstroms

    for idx, residue in enumerate(pdb.topology.residues()):
        for atom in residue.atoms():
            if atom.name.startswith("H"):
                if cgback_system.loc_types[idx] & LOC_TYPE_ENCODER["NTER"] == LOC_TYPE_ENCODER["NTER"] and atom.name == "H":
                    atom_type = HYDROGEN_ATOM_TYPE_ENCODER[residue.name]["H1"]
                else:
                    atom_type = HYDROGEN_ATOM_TYPE_ENCODER[residue.name][atom.name]
                cgback_system.hydrogen_coordinates[idx, atom_type] = minimized_positions[atom.index]
            else:
                atom_type = HEAVY_ATOM_TYPE_ENCODER[residue.name][atom.name]
                cgback_system.heavy_coordinates[idx, atom_type] = minimized_positions[atom.index]

