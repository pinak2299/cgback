import gc
import sys
import time
import random
import collections
import torch
import numpy as np
import logging
import pathlib
import getpass
import socket
import importlib.metadata as metadata
import importlib.resources as resources
from datetime import datetime
from torch.utils.data import Subset, DataLoader
from cgback.diffuser import DDPM, DDIM
from cgback.system import System, subsystem_from_indexes, update_system_from_subsystem
from cgback.hydrogen import update_hydrogen_coordinates, find_incorrect_chiral_centers, calculate_num_chiral_centers
from cgback.parser import system_from_pdb_path, system_from_cif_path, write_pdb_from_system, write_cif_from_system, BACKBONE_ATOM_TYPE_ENCODER, RESIDUE_TYPE_DECODER
from cgback.dataset import ProteinDataset
from cgback.penetration import find_residues_with_rings, build_ring_neighbor_list, find_penetrations, initialize_ring_descriptors, update_ring_descriptors, RingDescriptor
from cgback.clash import build_clash_neighbor_list, find_clashes
from cgback.dataloader import collate_fn
try:
    from cgback.minimization import energy_minimization
    openmm_enabled = True
except ImportError:
    energy_minimization = None
    openmm_enabled = False


class Backmapper:
    def __init__(self, args):
        # Step 1: Save arguments
        self.args = args

        # Step 2: Setup log
        self.logger = None
        self.console_handler = None
        self.logger_formatter = None
        self.setup_logger()

        # Step 3: Log run information
        self.log_run_info()

        # Step 4: Setup RNGs
        self.setup_rng_seed()

        # Step 5: Setup device
        self.device = torch.device(self.args.device)

        # Step 6: Setup model
        self.diffuser = None
        self.dim_hidden = None
        self.num_timesteps = None
        self.num_layers = None
        self.cutoff = None
        self.checkpoint_path = None
        self.model = None
        self.checkpoint = None
        if self.args.model == "C": self.load_c_model()
        if self.args.model == "S": self.load_s_model()
        if self.args.model == "M": self.load_m_model()
        if self.args.model == "L": self.load_l_model()

        # Step 7: Setup number of steps
        if self.args.num_timesteps is not None: self.set_model_timesteps(self.args.num_timesteps)

        # Step 8: Initialize timer variables
        self.setup_time = None
        self.sampling_time = None
        self.add_hydrogen_time = None
        self.fix_structure_time = None
        self.write_time = None
        self.minimization_time = None

    def log_run_info(self):
        self.logger.info(f"cgback {metadata.version(__package__)}")
        self.logger.info(f"User: {getpass.getuser()}@{socket.gethostname()}")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}")
        self.logger.info(f"Input: '{self.args.INPUT}'")
        self.logger.info(f"Output: '{self.args.output}'")
        self.logger.info(f"Batch size: {self.args.batch}")
        self.logger.info(f"Seed: {self.args.seed}")
        self.logger.info(f"Ignore existing atoms: {self.args.ignore_existing}")
        if self.args.skip_sampling: self.logger.info(f"Skip sampling: {self.args.skip_sampling}")
        if self.args.skip_add_hydrogen: self.logger.info(f"Skip adding hydrogen atoms: {self.args.skip_add_hydrogen}")
        if self.args.skip_fix_structure: self.logger.info(f"Skip fixing structure artifacts: {self.args.skip_fix_structure}")

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(self.console_handler)
        self.logger_formatter = logging.Formatter(fmt="[{levelname}] {message}", style="{")
        self.console_handler.setFormatter(self.logger_formatter)
        self.logger.setLevel("WARNING")
        if self.args.verbose: self.logger.setLevel("INFO")
        if self.args.debug: self.logger.setLevel("DEBUG")

    def setup_rng_seed(self):
        seed = self.args.seed

        random.seed(seed)
        self.logger.debug(f"random seed set to {seed}")

        np.random.seed(seed)
        self.logger.debug(f"numpy seed set to {seed}")

        torch.manual_seed(seed)
        self.logger.debug(f"torch seed set to {seed}")

        torch.cuda.manual_seed(seed)
        self.logger.debug(f"cuda seed set to {seed}")

    def unload_model(self):
        del self.model
        if self.args.device == "cuda": torch.cuda.empty_cache()
        gc.collect()

        self.num_timesteps = None
        self.num_layers = None
        self.cutoff = None
        self.checkpoint_path = None
        self.model = None
        self.checkpoint = None
        self.logger.debug("Diffusion model unloaded")

        self.logger.info("Unloaded model")

    def prepare_model(self):
        try:
            self.model = self.model.to(self.device)
            self.logger.info(f"Using device: {self.device}")
            self.logger.debug(f"model sent to device {self.device}")
        except RuntimeError:
            self.logger.error(f"Device '{self.device}' is not available")
            sys.exit(0)

        self.checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        self.logger.debug("checkpoint loaded")

        model_state_dict = collections.OrderedDict()
        for k, v in self.checkpoint["model"]["state_dict"].items():
            k = k[7:] if k.startswith("module.") else k
            model_state_dict[k] = v
        self.checkpoint["model"]["state_dict"] = model_state_dict
        self.logger.debug("checkpoint state dict reformatted")

        try:
            self.model.load_state_dict(self.checkpoint["model"]["state_dict"])
            self.logger.debug("checkpoint state dict loaded")
        except RuntimeError:
            self.logger.error("Model checkpoint is incompatible with the current version of CGBack")
            sys.exit(0)

    def load_model(self, num_timesteps: int, num_layers: int, dim_hidden: int, cutoff: float):
        self.num_timesteps = num_timesteps
        self.logger.debug(f"number of timesteps set to {self.num_timesteps}")

        self.dim_hidden = dim_hidden
        self.logger.debug(f"dimension of hidden layers set to {self.dim_hidden}")

        self.num_layers = num_layers
        self.logger.debug(f"number of layers set to {self.num_layers}")

        self.cutoff = cutoff
        self.logger.debug(f"cutoff set to {self.cutoff}")

        self.checkpoint_path = resources.files(__package__).joinpath("checkpoints").joinpath(f"protein-carbon-alpha-cutoff-{int(cutoff):03}-steps-{num_timesteps:03}-layers-{num_layers:03}-hidden-{dim_hidden:03}.pt")
        self.logger.debug(f"checkpoint path set to {self.checkpoint_path}")

        self.diffuser = self.args.diffuser
        self.logger.info(f"Diffuser set to {self.diffuser}")

        if self.diffuser == "DDPM":
            self.model = DDPM(self.num_timesteps, 60, dim_hidden, self.num_layers, "cosine")
            self.logger.debug("DDPM model initialized")
        if self.diffuser == "DDIM":
            self.model = DDIM(self.num_timesteps, 60, dim_hidden, self.num_layers, "cosine")
            self.logger.debug("DDIM model initialized")
        self.prepare_model()

    def load_s_model(self):
        self.load_model(20, 2, 128, 8.0)
        self.logger.info("Loaded model S")

    def load_m_model(self):
        self.load_model(20, 3, 128, 8.0)
        self.logger.info("Loaded model M")

    def load_l_model(self):
        self.load_model(20, 4, 128, 8.0)
        self.logger.info("Loaded model L")

    def load_c_model(self):
        if self.args.custom_model_num_timesteps:
            self.num_timesteps = self.args.custom_model_num_timesteps
            self.logger.debug(f"number of timesteps set to {self.num_timesteps}")
        else:
            return

        if self.args.custom_model_dim_hidden:
            self.dim_hidden = self.args.custom_model_dim_hidden
            self.logger.debug(f"dimension of hidden layers set to {self.dim_hidden}")
        else:
            return

        if self.args.custom_model_num_layers:
            self.num_layers = self.args.custom_model_num_layers
            self.logger.debug(f"number of layers set to {self.num_layers}")
        else:
            return

        if self.args.custom_model_cutoff:
            self.cutoff = self.args.custom_model_cutoff
            self.logger.debug(f"cutoff set to {self.cutoff}")
        else:
            return

        if self.args.custom_model_checkpoint_path:
            self.checkpoint_path = self.args.custom_model_checkpoint_path
            self.logger.debug(f"checkpoint path set to {self.checkpoint_path}")
        else:
            return

        self.diffuser = self.args.diffuser
        self.logger.info(f"Diffuser set to {self.diffuser}")

        if self.diffuser == "DDPM":
            self.model = DDPM(self.num_timesteps, 60, self.dim_hidden, self.num_layers, "cosine")
            self.logger.debug("DDPM model initialized")
        if self.diffuser == "DDIM":
            self.model = DDIM(self.num_timesteps, 60, self.dim_hidden, self.num_layers, "cosine")
            self.logger.debug("DDIM model initialized")

        self.prepare_model()

        self.logger.info(f"Custom model checkpoint path: {self.args.custom_model_checkpoint_path}")
        self.logger.info(f"Custom model number of time steps: {self.args.custom_model_num_timesteps}")
        self.logger.info(f"Custom model number of layers: {self.args.custom_model_num_layers}")
        self.logger.info(f"Custom model dimension of hidden layers: {self.args.custom_model_dim_hidden}")
        self.logger.info(f"Custom model cutoff: {self.args.custom_model_cutoff:.2f}")
        self.logger.info("Loaded model C")

    def set_model_timesteps(self, num_timesteps):
        if num_timesteps <= 0:
            self.logger.error("Number of steps must be positive")
            sys.exit(0)

        self.model.num_timesteps = num_timesteps
        self.logger.debug(f"number of timesteps set to {self.model.num_timesteps}")

        self.model = self.model.to(self.device)
        self.logger.debug(f"model sent to device {self.device}")

    def initialize_system(self) -> System:
        system = None
        if self.args.INPUT.endswith(".pdb"): system = system_from_pdb_path(self.args.INPUT, self.logger)
        if self.args.INPUT.endswith(".cif"): system = system_from_cif_path(self.args.INPUT, self.logger)
        if system is None:
            self.logger.error(f"Unknown input file format: {self.args.INPUT}")
            sys.exit(0)
        self.logger.debug(f"system initialized from {self.args.INPUT}")

        return system

    def denoise_graphs(self, dataloader):
        batches = []
        for idx, data in enumerate(dataloader):
            bar_description = f"[INFO] Denoising graphs [BATCH {idx + 1}/{len(dataloader)}]"
            data = self.model.sample(data, device=self.device, bar_description=bar_description, verbose=self.args.verbose)
            data.xw = data.xw.detach().cpu()
            data.num_h = data.num_h.detach().cpu()
            data.num_u = data.num_u.detach().cpu()
            batches.append(data)

        return batches

    @staticmethod
    def coordinates_from_batches(batches, dataset):
        # Initialize the number of residues to process
        num_residues = len(dataset)
        # Initialize positions
        coordinates = np.full((num_residues, 15, 3), np.nan)
        # Initialize residue index counter
        res_idx = 0
        # Iterate over batches
        for batch in batches:
            # Initialize offset to decode batched graphs
            offset = 0
            # Iterate over residues packed in a batch
            for num_h, num_u in zip(batch.num_h, batch.num_u):
                # Get positions of the current residue
                lower = offset
                upper = offset + num_h.item()
                # OXT is kept as NaN because it is added in later steps
                coordinates[res_idx, 0:1] = batch.xw[lower+0:lower+1].numpy() - batch.xw[upper+0:upper+1].numpy() # N
                coordinates[res_idx, 1:2] = batch.xw[upper+0:upper+1].numpy() - batch.xw[upper+0:upper+1].numpy() # CA
                coordinates[res_idx, 2:4] = batch.xw[lower+1:lower+3].numpy() - batch.xw[upper+0:upper+1].numpy() # C, O
                coordinates[res_idx, 5:5+(upper-lower-3)] = batch.xw[lower+3:upper].numpy() - batch.xw[upper+0:upper+1].numpy() # R
                # Update residue index counter
                res_idx += 1
                # Update offset
                offset += num_h.item() + num_u.item()

        coordinates = coordinates

        return coordinates

    def setup_system(self) -> System:
        try:
            system = self.initialize_system()
        except FileNotFoundError:
            self.logger.error(f"No such file or directory: '{self.args.INPUT}'")
            sys.exit(0)

        if self.args.ignore_existing:
            system.heavy_coordinates[:, :BACKBONE_ATOM_TYPE_ENCODER["CA"], :] = np.nan
            system.heavy_coordinates[:, BACKBONE_ATOM_TYPE_ENCODER["CA"]+1:, :] = np.nan
            system.hydrogen_coordinates.fill(np.nan)
            system.heavy_mask.fill(True)
            system.hydrogen_mask.fill(True)

        return system

    def setup_data(self, system: System) -> tuple[Subset, DataLoader]:
        dataset = self.build_dataset(system)
        sampling_indexes = np.flatnonzero(system.heavy_mask).tolist()
        dataset = Subset(dataset, sampling_indexes)
        dataloader = self.build_dataloader(dataset)

        return dataset, dataloader

    def sampling(self, system: System, dataset, dataloader, suffix: str | None = "#1-SAMPLING"):
        batches = self.denoise_graphs(dataloader)
        new_coordinates = self.coordinates_from_batches(batches, dataset)
        sampling_indexes = np.flatnonzero(system.heavy_mask).tolist()
        new_coordinates += system.heavy_coordinates[sampling_indexes, 1:2, :]
        system.heavy_coordinates[sampling_indexes] = new_coordinates
        self.logger.info("Sampling finished")

        if self.args.keep and suffix is not None: self.write_system(system, suffix)

    def add_hydrogen_atoms(self, system: System, suffix: str | None = "#2-ADD-HYDROGEN"):
        update_hydrogen_coordinates(system)
        self.logger.info("Adding hydrogen atoms finished")

        if self.args.keep and suffix is not None: self.write_system(system, suffix)

    def fix_structure(self, system: System):
        # Step 1: setup model
        self.unload_model()
        if self.args.fix_structure_model == "S": self.load_s_model()
        if self.args.fix_structure_model == "M": self.load_m_model()
        if self.args.fix_structure_model == "L": self.load_l_model()
        dataset = ProteinDataset(system, self.cutoff)

        # Step 2: calculate indexes of residues with rings
        ring_residue_idxs = find_residues_with_rings(system)
        self.logger.info(f"Number of residues with rings: {len(ring_residue_idxs)}")

        # Step 3: calculate the number of chiral centers
        num_chiral_centers = calculate_num_chiral_centers(system)
        self.logger.info(f"Number of chiral centers: {num_chiral_centers}")

        # Step 4: calculate pairs of possible clahses
        neighbor_clash_list = build_clash_neighbor_list(system)

        # Step 4: calculate pairs of possible penetrations
        ring_neighbor_list = build_ring_neighbor_list(system, ring_residue_idxs)

        # Step 5: initialize ring descriptors
        ring_descriptors = initialize_ring_descriptors(system, ring_residue_idxs)

        # Step 6: initialize masks
        fix_mask = np.full(len(system), False, dtype=np.bool)

        # Step 7: initialize counter
        iterations = 0

        # Step 8: save structure at current iteration
        if self.args.keep: self.write_system(system, f"#3-FIX-STRUCTURE-ITERATION-{iterations}")

        # Step 9: solve penetrations
        while True:
            # Step 9.1: find penetrations by looping over all the possible pairs
            penetrations = find_penetrations(system, ring_descriptors, ring_neighbor_list)
            self.report_penetrations(system, ring_descriptors, penetrations)
            for penetration in penetrations:
                idx = ring_descriptors[penetration[0]].residue_index
                fix_mask[idx] = True
                idx = penetration[1]
                fix_mask[idx] = True

            # Step 9.2: find incorrect chiralities
            incorrect_chiral_centers, num_chiral_centers_with_wrong_chirality = find_incorrect_chiral_centers(system)
            incorrect_chiral_centers_indexes = [index for index, _ in incorrect_chiral_centers]
            self.report_incorrect_chiral_centers(system, incorrect_chiral_centers)
            for idx in incorrect_chiral_centers_indexes:
                fix_mask[idx] = True

            # Step 9.3: fine clashes
            clashes = None
            if not self.args.skip_fix_structure_clashes:
                clashes = find_clashes(system, neighbor_clash_list)
                self.report_clashes(system, clashes)
                for clash in clashes:
                    fix_mask[clash[0]] = True
                    fix_mask[clash[1]] = True

            # Step 9.4: report scores
            num_residues_with_ring_penetration = len(set([ring_descriptors[item[0]].residue_index for item in penetrations]))
            num_residues_with_clashes = None
            if not self.args.skip_fix_structure_clashes: num_residues_with_clashes = len(set([item for pair in clashes for item in pair]))
            num_residues_with_ring = len(ring_residue_idxs)
            num_residues = len(system)
            if num_residues_with_ring > 0:
                ring_score = num_residues_with_ring_penetration / num_residues_with_ring * 100.0
            else:
                ring_score = 0.0
            clash_score = None
            if not self.args.skip_fix_structure_clashes: clash_score = num_residues_with_clashes / num_residues * 100.0
            if num_chiral_centers > 0:
                chirality_score = num_chiral_centers_with_wrong_chirality / num_chiral_centers * 100.0
            else:
                chirality_score = 0.0
            line = ""
            line += f"Iteration: {iterations:6}"
            line += f" | Ring score: {ring_score:6.2f}%"
            if not self.args.skip_fix_structure_clashes: line += f" | Clash score: {clash_score:6.2f}%"
            line += f" | Chirality score: {chirality_score:6.2f}%"
            line += f" | Bad rings: {num_residues_with_ring_penetration:7}"
            if not self.args.skip_fix_structure_clashes: line += f" | Bad residues: {num_residues_with_clashes:7}"
            line += f" | Bad centers: {num_chiral_centers_with_wrong_chirality:7}"
            self.logger.info(line)

            # Step 9.5: merge lists
            bad_indexes = np.where(fix_mask)[0].tolist()

            # Step 9.6: stop if maximum number of iterations was reached
            if len(bad_indexes) == 0:
                self.logger.info("Fixing structure artifacts finished")
                break

            # Step 9.7: stop if maximum number of iterations was reached
            if iterations == self.args.fix_structure_max_iterations:
                self.logger.warning("Maximum number of iterations reached")
                self.logger.warning(f"Number of unsolved penetrations: {len(penetrations)}")
                self.logger.warning(f"Number of unsolved chirality centers: {len(incorrect_chiral_centers_indexes)}")
                break

            # Step 9.8: update indexes
            masked_bad_indexes = [index for index in bad_indexes if system.heavy_mask[index]]
            if len(masked_bad_indexes) == 0:
                masked_bad_indexes = bad_indexes
                for index in masked_bad_indexes: system.heavy_mask[index] = True
                self.report_regeneration_of_existing_residues(system, masked_bad_indexes)

            # Step 9.9: regenerate residues
            self.logger.info(f"Regenerating {len(bad_indexes)} residues")
            new_system = subsystem_from_indexes(system, masked_bad_indexes)
            new_coords = np.random.normal(0.0, 1.0, size=new_system.heavy_coordinates.shape)
            new_system.heavy_coordinates[:,:BACKBONE_ATOM_TYPE_ENCODER["CA"],:] = new_coords[:,:BACKBONE_ATOM_TYPE_ENCODER["CA"],:]
            new_system.heavy_coordinates[:,BACKBONE_ATOM_TYPE_ENCODER["CA"]+1:,:] = new_coords[:,BACKBONE_ATOM_TYPE_ENCODER["CA"]+1:,:]
            new_dataset = Subset(dataset, masked_bad_indexes)
            new_dataloader = DataLoader(new_dataset, self.args.batch, shuffle=False, drop_last=False, collate_fn=collate_fn)
            self.sampling(new_system, new_dataset, new_dataloader, suffix=None)

            # Step 9.10: update the system with new residues
            system = update_system_from_subsystem(system, new_system, masked_bad_indexes)
            if not (self.args.skip_add_hydrogen or np.isnan(system.hydrogen_coordinates).all()): self.add_hydrogen_atoms(system, suffix=None)

            # Step 9.11: update ring descriptors
            update_ring_descriptors(system, ring_descriptors, fix_mask)

            # Step 9.12: update mask
            fix_mask.fill(False)

            # Step 9.13: update iteration counter
            iterations += 1

            # Step 9.14: structure at current iteration
            if self.args.keep:
                if self.args.fix_structure_max_iterations < 10:
                    self.write_system(system, f"#3-FIX-STRUCTURE-ITERATION-{iterations:01}")
                elif self.args.fix_structure_max_iterations < 100:
                    self.write_system(system, f"#3-FIX-STRUCTURE-ITERATION-{iterations:02}")
                elif self.args.fix_structure_max_iterations < 1000:
                    self.write_system(system, f"#3-FIX-STRUCTURE-ITERATION-{iterations:03}")
                elif self.args.fix_structure_max_iterations < 10000:
                    self.write_system(system, f"#3-FIX-STRUCTURE-ITERATION-{iterations:04}")
                else:
                    self.write_system(system, f"#3-FIX-STRUCTURE-ITERATION-{iterations}")


    def report_penetrations(self, system: System, ring_descriptors: list[RingDescriptor], penetration_residues: list[tuple[int,int]]):
        for idx1, idx2 in penetration_residues:
            idx1 = ring_descriptors[idx1].residue_index
            seq1 = RESIDUE_TYPE_DECODER[system.residue_types[idx1]]
            seq2 = RESIDUE_TYPE_DECODER[system.residue_types[idx2]]
            self.logger.info(f"Penetration found between {seq1}:{idx1} and {seq2}:{idx2}")

    def report_clashes(self, system: System, clashes: list[tuple[int,int]]) -> None:
        for idx0, idx1 in clashes:
            seq0 = RESIDUE_TYPE_DECODER[system.residue_types[idx0]]
            seq1 = RESIDUE_TYPE_DECODER[system.residue_types[idx1]]
            self.logger.info(f"Clash between {seq0}:{idx0} and {seq1}:{idx1}")

    def report_incorrect_chiral_centers(self, system: System, incorrect_chiral_centers: list[tuple[int,str]]) -> None:
        for idx, center in incorrect_chiral_centers:
            seq = RESIDUE_TYPE_DECODER[system.residue_types[idx]]
            self.logger.info(f"Incorrect chirality in {seq}:{idx}:{center}")

    def report_regeneration_of_existing_residues(self, system: System, indexes: list[int]):
        for index in indexes:
            seq = RESIDUE_TYPE_DECODER[system.residue_types[index]]
            self.logger.info(f"Regenerating existing residue {seq}:{index}")

    def energy_minimization(self, system: System):
        if not openmm_enabled:
            self.logger.warning("Please install cgback with OpenMM support to perform energy minimization")
            return
        if self.args.skip_add_hydrogen:
            self.logger.warning("Energy minimization requires hydrogen atoms to be added")
            return
        self.logger.info("Performing energy minimization")
        try:
            energy_minimization(
                system,
                self.args.energy_minimization_cutoff,
                self.args.energy_minimization_max_iterations,
                self.args.energy_minimization_tolerance,
                self.args.energy_minimization_log_interval,
                self.args.energy_minimization_ignore_existing,
                self.logger
            )
        except Exception as e:
            self.logger.error(f"Energy minimization failed: {e}")
            sys.exit(0)
        self.logger.info("Energy minimization finished")

    def write_system(self, system: System, suffix: str = ""):
        if self.args.output.endswith(".pdb"):
            pdb_path = pathlib.Path(f"{self.args.output}{suffix}")
            write_pdb_from_system(pdb_path, system)
        elif self.args.output.endswith(".cif"):
            cif_path = pathlib.Path(f"{self.args.output}{suffix}")
            write_cif_from_system(cif_path, system)
        else:
            self.logger.error("Unknown output file format: " + self.args.output)

    def build_dataset(self, system: System) -> ProteinDataset:
        self.logger.info("Building dataset")
        dataset = ProteinDataset(system, self.cutoff)
        self.logger.info("Dataset built")
        self.logger.info(f"Number of graphs in the dataset: {len(dataset)}")

        return dataset

    def build_dataloader(self, dataset: ProteinDataset | Subset) -> DataLoader:
        self.logger.info("Building dataloader")
        dataloader = DataLoader(dataset, self.args.batch, shuffle=False, drop_last=False, collate_fn=collate_fn)
        self.logger.info("Dataloader built")

        return dataloader

    def write_profiler(self):
        total_time = self.setup_time + self.sampling_time + self.add_hydrogen_time + self.fix_structure_time + self.write_time + self.minimization_time
        setup_time_percentage = self.setup_time / total_time * 100
        sampling_time_percentage = self.sampling_time / total_time * 100
        add_hydrogen_time_percentage = self.add_hydrogen_time / total_time * 100
        fix_structure_time_percentage = self.fix_structure_time / total_time * 100
        write_time_percentage = self.write_time / total_time * 100
        minimization_time_percentage = self.minimization_time / total_time * 100
        self.logger.info(f"Time for setup:            {self.setup_time:>9.2f} s ({setup_time_percentage:>6.2f}%)")
        self.logger.info(f"Time for sampling:         {self.sampling_time:>9.2f} s ({sampling_time_percentage:>6.2f}%)")
        self.logger.info(f"Time for add hydrogen:     {self.add_hydrogen_time:>9.2f} s ({add_hydrogen_time_percentage:>6.2f}%)")
        self.logger.info(f"Time for fixing structure: {self.fix_structure_time:>9.2f} s ({fix_structure_time_percentage:>6.2f}%)")
        if self.args.energy_minimization: self.logger.info(f"Time for minimization:     {self.minimization_time:>9.2f} s ({minimization_time_percentage:>6.2f}%)")
        self.logger.info(f"Time for writing output:   {self.write_time:>9.2f} s ({write_time_percentage:>6.2f}%)")
        self.logger.info(f"Total time:                {total_time:>9.2f} s ({100.0:>6.2f}%)")

    def run(self):
        # Setup
        s = time.time()
        system = self.setup_system()
        dataset = None
        dataloader = None
        if not self.args.skip_sampling: dataset, dataloader = self.setup_data(system)
        self.setup_time = time.time() - s

        # Denoise graphs
        s = time.time()
        if not self.args.skip_sampling: self.sampling(system, dataset, dataloader)
        self.sampling_time = time.time() - s

        # Add hydrogen atoms
        s = time.time()
        if not self.args.skip_add_hydrogen: self.add_hydrogen_atoms(system)
        self.add_hydrogen_time = time.time() - s

        # Fix structure
        s = time.time()
        if not self.args.skip_fix_structure: self.fix_structure(system)
        self.fix_structure_time = time.time() - s

        # Energy minimization
        s = time.time()
        if self.args.energy_minimization: self.energy_minimization(system)
        self.minimization_time = time.time() - s

        # Write output
        s = time.time()
        self.write_system(system)
        self.write_time = time.time() - s

        # Write profiler
        self.write_profiler()
