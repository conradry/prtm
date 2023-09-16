from __future__ import annotations

import numpy as np
import py3Dmol
from matplotlib import colormaps, colors

from proteome.pdb_utils import overwrite_b_factors


def view_protein_with_bfactors(
    structure: "protein.ProteinBase",
    cmap: str,
    show_sidechains: bool = True,
) -> py3Dmol.view:
    band_edges = np.arange(50, 110, step=10)

    # Must be a 37 atom protein
    structure = structure.to_protein37()

    # If no atoms beyond the fifth are present then
    # we don't have sidechains to show
    if not np.any(structure.atom_mask[:, 5:]):
        show_sidechains = False

    # Get the colors from pyplot
    n = len(band_edges)
    color_map = colormaps.get_cmap(cmap).resampled(n)
    if isinstance(color_map, colors.LinearSegmentedColormap):
        band_colors = color_map(np.arange(0, color_map.N))
    else:
        band_colors = color_map.colors

    band_colors_hex = [colors.to_hex(c) for c in band_colors]

    # Bin the b-factors into one of the bands
    banded_b_factors = []
    mean_b_factors = (
        np.ma.masked_array(data=structure.b_factors, mask=(structure.atom_mask < 0.5))
        .mean(axis=1)
        .data
    )
    banded_b_factors = (
        np.digitize(mean_b_factors, band_edges, right=True)[:, None]
        * structure.atom_mask
    )

    # Update the b-factors in the PDB string to be band indices
    to_visualize_pdb = overwrite_b_factors(structure.to_pdb(), banded_b_factors)

    view = py3Dmol.view(width=800, height=600)
    view.addModelsAsFrames(to_visualize_pdb)
    color_map = {i: band_colors_hex[i] for i in range(len(band_edges))}
    style = {"cartoon": {"colorscheme": {"prop": "b", "map": color_map}}}

    if show_sidechains:
        style["stick"] = {}

    view.setStyle({"model": -1}, style)
    view.zoomTo()
    return view
