import dataclasses
from numpy.typing import NDArray


@dataclasses.dataclass
class System:
    chain_ids: NDArray
    residue_types: NDArray
    loc_types: NDArray
    heavy_coordinates: NDArray
    hydrogen_coordinates: NDArray
    heavy_mask: NDArray
    hydrogen_mask: NDArray

    def __len__(self):
        return len(self.residue_types)


def subsystem_from_indexes(system: System, indexes: list[int]) -> System:
    return System(
        system.chain_ids[indexes],
        system.residue_types[indexes],
        system.loc_types[indexes],
        system.heavy_coordinates[indexes],
        system.hydrogen_coordinates[indexes],
        system.heavy_mask[indexes],
        system.hydrogen_mask[indexes],
    )


def update_system_from_subsystem(system: System, subsystem: System, indexes: list[int]) -> System:
    system.chain_ids[indexes] = subsystem.chain_ids
    system.residue_types[indexes] = subsystem.residue_types
    system.loc_types[indexes] = subsystem.loc_types
    system.heavy_coordinates[indexes] = subsystem.heavy_coordinates
    system.hydrogen_coordinates[indexes] = subsystem.hydrogen_coordinates
    system.heavy_mask[indexes] = subsystem.heavy_mask
    system.hydrogen_mask[indexes] = subsystem.hydrogen_mask

    return system
