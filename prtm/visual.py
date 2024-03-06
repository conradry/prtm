from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import nglview as nv
import numpy as np
import py3Dmol
from matplotlib import colormaps, colors
from prtm.pdb_utils import overwrite_b_factors


def make_visualization_pdb(
    structure: "protein.ProteinBase",
    cmap: str,
    bfactor_is_confidence: bool = False,
) -> Tuple[str, Dict[int, str]]:
    """Creates a PDB string with b-factors mapped to a color scheme.

    Args:
        structure: The protein structure to render.
        bfactor_is_confidence: Whether to treat the b-factors as confidence scores.
        cmap: The name of the matplotlib colormap to use.

    Returns:
        pdb_str: A PDB string with b-factors mapped to a color scheme.
        color_map: A dictionary mapping band indices to hex colors.
    """
    if bfactor_is_confidence:
        band_edges = np.arange(50, 110, step=10)
    else:
        band_edges = np.linspace(0, structure.b_factors.max(), num=10)

    # Must be a 37 atom protein
    structure = structure.to_protein37()

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
    color_map = {i: band_colors_hex[i] for i in range(len(band_edges))}

    # Update the b-factors in the PDB string to be band indices
    to_viz_pdb = overwrite_b_factors(structure.to_pdb(write_hetatoms=False), banded_b_factors)

    return to_viz_pdb, color_map


def view_protein_with_bfactors(
    structure: "protein.ProteinBase",
    cmap: str,
    bfactor_is_confidence: bool = False,
    show_sidechains: bool = True,
) -> py3Dmol.view:
    """Renders a protein structure with b-factors mapped to a color scheme.

    Args:
        structure: The protein structure to render.
        cmap: The name of the matplotlib colormap to use.
        bfactor_is_confidence: Whether to treat the b-factors as confidence scores.
        show_sidechains: Whether to show the sidechains of the protein.
        Ignored if the protein has no sidechains.

    Returns:
        A py3Dmol.view object with the protein rendered.
    """

    to_visualize_pdb, color_map = make_visualization_pdb(
        structure, cmap, bfactor_is_confidence
    )

    view = py3Dmol.view(width=800, height=600)
    view.addModelsAsFrames(to_visualize_pdb)
    style = {"cartoon": {"colorscheme": {"prop": "b", "map": color_map}}}

    if show_sidechains and np.any(structure.atom_mask[:, 5:]):
        style["stick"] = {}

    view.setStyle({"model": -1}, style)
    view.zoomTo()
    return view


def view_aligned_structures_grid(
    structures: List["protein.ProteinBase"],
    cmap: str = "viridis",
    max_grid_cols: int = 5,
    bfactor_is_confidence: bool = True,
    show_sidechains: bool = True,
):
    # Make sure the all structures have the same sequence length
    assert (
        len(set([len(s.sequence()) for s in structures])) == 1
    ), "All structures must have the same sequence to be aligned!"

    # Align all structures to the first one
    aligned_structures = [structures[0]]
    for structure in structures[1:]:
        aligned_structures.append(aligned_structures[0].superimpose(structure))

    # Figure out the grid shape to fit everything
    n_structures = len(structures)
    grid_shape = (
        1 + (n_structures // max_grid_cols),
        min(n_structures, max_grid_cols),
    )
    view = py3Dmol.view(width=800, height=600, linked=True, viewergrid=grid_shape)

    for y in range(grid_shape[0]):
        for x in range(grid_shape[1]):
            i = y * grid_shape[1] + x
            if i >= n_structures:
                break

            structure = aligned_structures[i]
            to_visualize_pdb, color_map = make_visualization_pdb(
                structure, cmap, bfactor_is_confidence
            )

            view.addModelsAsFrames(to_visualize_pdb, viewer=(y, x))
            style = {"cartoon": {"colorscheme": {"prop": "b", "map": color_map}}}

            if show_sidechains and np.any(structure.atom_mask[:, 5:]):
                style["stick"] = {}

            view.setStyle({"model": -1}, style, viewer=(y, x))
            view.zoomTo(viewer=(y, x))

    return view.render()


def view_superimposed_structures(
    structure1: "protein.ProteinBase",
    structure2: "protein.ProteinBase",
    color1: str = "red",
    color2: str = "blue",
):
    # Make sure the all structures have the same sequence
    assert len(structure1.sequence()) == len(
        structure2.sequence()
    ), "Both structures must have the same sequence to be aligned!"

    structure2 = structure1.superimpose(structure2)

    view = py3Dmol.view(width=800, height=600)
    view.addModelsAsFrames(structure1.to_pdb())
    view.addModelsAsFrames(structure2.to_pdb())
    style1 = {"cartoon": {"color": color1, "opacity": 0.8}}
    style2 = {"cartoon": {"color": color2}}

    # To cluttered to show sidechains
    view.setStyle({"model": 0}, style1)
    view.setStyle({"model": 1}, style2)
    view.zoomTo()
    return view


def view_ca_trace(
    structure: "protein.ProteinCATrace", color: str = "red"
) -> nv.NGLWidget:
    """Renders a protein CA trace with a specified color."""
    view = nv.NGLWidget()
    view.add_structure(nv.BiopythonStructure(structure.to_biopdb_structure()))
    view.clear()
    view.add_cartoon(selection="protein", color=color)
    view.center()
    return view


def view_superimposed_ca_traces(
    structures: List["protein.ProteinCATrace"],
) -> nv.NGLWidget:
    """Renders superimposed protein CA traces with default colors."""
    # Make sure the all structures have the same sequence
    assert (
        len(set([len(s.sequence()) for s in structures])) == 1
    ), "All structures must have the same sequence to be aligned!"

    # Align the structures to the first one in the list
    view = nv.NGLWidget()
    ref_structure = structures[0]
    view.add_structure(nv.BiopythonStructure(ref_structure.to_biopdb_structure()))
    for i, mov_structure in enumerate(structures[1:], 1):
        aligned_structure = ref_structure.superimpose(mov_structure)
        view.add_structure(
            nv.BiopythonStructure(aligned_structure.to_biopdb_structure())
        )

    view.center()

    return view
