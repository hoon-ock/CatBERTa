"""A class for analyzing the nature of the site and how the adsorbate is bound."""

import numpy as np
from ase import neighborlist, Atoms
from ase.neighborlist import natural_cutoffs
from scipy.spatial.distance import euclidean
from itertools import combinations


class SiteAnalyzer:
    def __init__(self, adslab, cutoff_multiplier=1.0):
        """
        Initialize class to handle site based analysis.

        Args:
            adslab (ase.Atoms): object of the slab with the adsorbate placed.
        """
        self.atoms = adslab
        self.cutoff_multiplier = cutoff_multiplier
        self.binding_info = self._find_binding_graph()

    def _find_binding_graph(self):
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()

        adsorbate_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 2]
        slab_atom_idxs = [idx for idx, tag in enumerate(tags) if tag != 2]

        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)

        binding_info = []
        adslab_positions = self.atoms.get_positions()
        for idx in adsorbate_atom_idxs:
            if sum(connectivity[idx][slab_atom_idxs]) >= 1:
                bound_slab_idxs = [
                    idx_slab
                    for idx_slab in slab_atom_idxs
                    if connectivity[idx][idx_slab] == 1
                ]
                ads_idx_info = {
                    "adsorbate_idx": idx,
                    "adsorbate_element": elements[idx],
                    "slab_atom_elements": [
                        element
                        for idx_el, element in enumerate(elements)
                        if idx_el in bound_slab_idxs
                    ],
                    "slab_atom_idxs": bound_slab_idxs,
                    "bound_position": adslab_positions[idx],
                }
                binding_info.append(ads_idx_info)
        return binding_info

    def _get_connectivity(self, atoms, cutoff_multiplier=1.0):
        """
        Note: need to condense this with the surface method
        Generate the connectivity of an atoms obj.
        Args:
            atoms (ase.Atoms): object which will have its connectivity considered
            cutoff_multiplier (float, optional): cushion for small atom movements when assessing
                atom connectivity
        Returns:
            (np.ndarray): The connectivity matrix of the atoms object.
        """
        cutoff = natural_cutoffs(atoms, mult=cutoff_multiplier)
        neighbor_list = neighborlist.NeighborList(
            cutoff,
            self_interaction=False,
            bothways=True,
            skin=0.05,
        )
        neighbor_list.update(atoms)
        matrix = neighborlist.get_connectivity_matrix(neighbor_list.nl).toarray()
        return matrix

    def get_dentate(self):
        """
        Get the number of adsorbate atoms that are bound to the surface.

        Returns:
            (int): The number of binding interactions
        """
        return len(self.binding_info)

    def get_site_types(self):
        """
        Get the number of surface atoms the bound adsorbate atoms are interacting with as a
        proximate for hollow, bridge, and atop binding.

        Returns:
            (list[int]): number of interacting surface atoms for each adsorbate atom bound.
        """
        return [len(binding["slab_atom_idxs"]) for binding in self.binding_info]

    def get_bound_atom_positions(self):
        """
        Get the euclidean coordinates of all bound adsorbate atoms.

        Returns:
            (list[np.array]): euclidean coordinates of bound atoms
        """
        positions = []
        for atom in self.binding_info:
            positions.append(atom["bound_position"])
        return positions

    def get_minimum_site_proximity(self, site_to_compare):
        """
        Note: might be good to check the surfaces are identical and raise an error otherwise.
        Get the minimum distance between bound atoms on the surface between two adsorbates.

        Args:
            site_to_compare (catapalt.SiteAnalyzer): site analysis instance of the other adslab.

        Returns:
            (float): The minimum distance between bound adsorbate atoms on a surface.
                and returns `np.nan` if one or more of the configurations was not
                surface bound.
        """
        this_positions = self.get_bound_atom_positions()
        other_positions = site_to_compare.get_bound_atom_positions()
        distances = []
        if len(this_positions) > 0 and len(other_positions) > 0:
            for this_position in this_positions:
                for other_position in other_positions:
                    fake_atoms = Atoms("CO", positions=[this_position, other_position])
                    distances.append(fake_atoms.get_distance(0, 1, mic=True))
            return min(distances)
        else:
            return np.nan

    def get_adsorbate_bond_lengths(self):
        """ """
        bond_lengths = {"adsorbate-adsorbate": {}, "adsorbate-surface": {}}
        adsorbate = self.atoms[
            [idx for idx, tag in enumerate(self.atoms.get_tags()) if tag == 2]
        ]
        adsorbate_connectivity = self._get_connectivity(adsorbate)
        combos = list(combinations(range(len(adsorbate)), 2))
        for combo in combos:
            if adsorbate_connectivity[combo[0], combo[1]] == 1:
                bond_lengths["adsorbate-adsorbate"][
                    tuple(np.sort(combo))
                ] = adsorbate.get_distance(combo[0], combo[1], mic=True)

        for ads_info in self.binding_info:
            adsorbate_idx = ads_info["adsorbate_idx"]
            bond_lengths["adsorbate-surface"][adsorbate_idx] = [
                self.atoms.get_distances(adsorbate_idx, slab_idx, mic=True)[0]
                for slab_idx in ads_info["slab_atom_idxs"]
            ]

        return bond_lengths