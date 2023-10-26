import random

import numpy as np
import torch
from prtm import protein

try:
    import pyrosetta

    pyrosetta.init(silent=True, extra_options="-mute all")
    APPROX = False
except:
    print("WARNING: pyRosetta not found, will use an approximate SSE calculation")
    APPROX = True


def extract_secstruc(structure: protein.ProteinBase):
    idx = structure.residue_index
    if APPROX:
        aa_sequence = structure.aatype
        secstruct = get_sse(structure.atom_positions[:, 1])
    else:
        dssp = pyrosetta.rosetta.core.scoring.dssp
        pose = structure.to_rosetta_pose()
        dssp.Dssp(pose).insert_ss_into_pose(pose, True)
        aa_sequence = pose.sequence()
        secstruct = pose.secstruct()

    secstruc_dict = {
        "sequence": [i for i in aa_sequence],
        "idx": [int(i) for i in idx],
        "ss": [i for i in secstruct],
    }
    return secstruc_dict


def ss_to_tensor(ss):
    """
    Function to convert ss files to indexed tensors
    0 = Helix
    1 = Strand
    2 = Loop
    3 = Mask/unknown
    4 = idx for pdb
    """
    ss_conv = {"H": 0, "E": 1, "L": 2}
    idx = np.array(ss["idx"])
    ss_int = np.array([int(ss_conv[i]) for i in ss["ss"]])
    return ss_int, idx


def mask_ss(ss, idx, min_mask=0, max_mask=1.0):
    mask_prop = random.uniform(min_mask, max_mask)
    transitions = np.where(ss[:-1] - ss[1:] != 0)[
        0
    ]  # gets last index of each block of ss
    stuck_counter = 0
    while len(ss[ss == 3]) / len(ss) < mask_prop or stuck_counter > 100:
        width = random.randint(1, 9)
        start = random.choice(transitions)
        offset = random.randint(-8, 1)
        try:
            ss[start + offset : start + offset + width] = 3
        except:
            stuck_counter += 1
            pass
    ss = torch.tensor(ss)
    ss = torch.nn.functional.one_hot(ss, num_classes=4)
    ss = torch.cat((ss, torch.tensor(idx)[..., None]), dim=-1)
    #     mask = torch.where(torch.argmax(ss[:,:-1], dim=-1) == 3, False, True)
    mask = torch.tensor(np.where(np.argmax(ss[:, :-1].numpy(), axis=-1) == 3))
    return ss, mask


def generate_Cbeta(N, Ca, C):
    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    # Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    # fd: below matches sidechain generator (=Rosetta params)
    Cb = -0.57910144 * a + 0.5689693 * b - 0.5441217 * c + Ca

    return Cb


def get_pair_dist(a, b):
    """calculate pair distances between two sets of points

    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist


def construct_block_adj_matrix(sstruct, xyz, cutoff=6, include_loops=False):
    """
    Given a sstruct specification and backbone coordinates, build a block adjacency matrix.

    Input:

        sstruct (torch.FloatTensor): (L) length tensor with numeric encoding of sstruct at each position

        xyz (torch.FloatTensor): (L,3,3) tensor of Cartesian coordinates of backbone N,Ca,C atoms

        cutoff (float): The Cb distance cutoff under which residue pairs are considered adjacent
                        By eye, Nate thinks 6A is a good Cb distance cutoff

    Output:

        block_adj (torch.FloatTensor): (L,L) boolean matrix where adjacent secondary structure contacts are 1
    """

    L = xyz.shape[0]

    # three anchor atoms
    N = xyz[:, 0]
    Ca = xyz[:, 1]
    C = xyz[:, 2]

    # recreate Cb given N,Ca,C
    Cb = generate_Cbeta(N, Ca, C)

    # May need a batch dimension - NRB
    dist = get_pair_dist(Cb, Cb)  # [L,L]
    dist[torch.isnan(dist)] = 999.9

    dist += 999.9 * torch.eye(L, device=xyz.device)
    # Now we have dist matrix and sstruct specification, turn this into a block adjacency matrix
    # There is probably a way to do this in closed-form with a beautiful einsum but I am going to do the loop approach

    # First: Construct a list of segments and the index at which they begin and end
    segments = []

    begin = -1
    end = -1

    for i in range(sstruct.shape[0]):
        # Starting edge case
        if i == 0:
            begin = 0
            continue

        if not sstruct[i] == sstruct[i - 1]:
            end = i
            segments.append((sstruct[i - 1], begin, end))

            begin = i

    # Ending edge case: last segment is length one
    if not end == sstruct.shape[0]:
        segments.append((sstruct[-1], begin, sstruct.shape[0]))

    block_adj = torch.zeros_like(dist)
    for i in range(len(segments)):
        curr_segment = segments[i]

        if curr_segment[0] == 2 and not include_loops:
            continue

        begin_i = curr_segment[1]
        end_i = curr_segment[2]
        for j in range(i + 1, len(segments)):
            j_segment = segments[j]

            if j_segment[0] == 2 and not include_loops:
                continue

            begin_j = j_segment[1]
            end_j = j_segment[2]

            if torch.any(dist[begin_i:end_i, begin_j:end_j] < cutoff):
                # Matrix is symmetic
                block_adj[begin_i:end_i, begin_j:end_j] = torch.ones(
                    end_i - begin_i, end_j - begin_j
                )
                block_adj[begin_j:end_j, begin_i:end_i] = torch.ones(
                    end_j - begin_j, end_i - begin_i
                )
    return block_adj


def get_sse(ca_coord):
    """
    calculates the SSE of a peptide chain based on the P-SEA algorithm (Labesse 1997)
    code borrowed from biokite: https://github.com/biokit/biokit
    """

    def vector_dot(v1, v2):
        return (v1 * v2).sum(-1)

    def norm_vector(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def displacement(atoms1, atoms2):
        v1 = np.asarray(atoms1)
        v2 = np.asarray(atoms2)
        if len(v1.shape) <= len(v2.shape):
            diff = v2 - v1
        else:
            diff = -(v1 - v2)
        return diff

    def distance(atoms1, atoms2):
        diff = displacement(atoms1, atoms2)
        return np.sqrt(vector_dot(diff, diff))

    def angle(atoms1, atoms2, atoms3):
        v1 = norm_vector(displacement(atoms1, atoms2))
        v2 = norm_vector(displacement(atoms3, atoms2))
        return np.arccos(vector_dot(v1, v2))

    def dihedral(atoms1, atoms2, atoms3, atoms4):
        v1 = norm_vector(displacement(atoms1, atoms2))
        v2 = norm_vector(displacement(atoms2, atoms3))
        v3 = norm_vector(displacement(atoms3, atoms4))

        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)

        # Calculation using atan2, to ensure the correct sign of the angle
        x = vector_dot(n1, n2)
        y = vector_dot(np.cross(n1, n2), v2)
        return np.arctan2(y, x)

    _radians_to_angle = 2 * np.pi / 360

    _r_helix = ((89 - 12) * _radians_to_angle, (89 + 12) * _radians_to_angle)
    _a_helix = ((50 - 20) * _radians_to_angle, (50 + 20) * _radians_to_angle)
    _d2_helix = ((5.5 - 0.5), (5.5 + 0.5))
    _d3_helix = ((5.3 - 0.5), (5.3 + 0.5))
    _d4_helix = ((6.4 - 0.6), (6.4 + 0.6))

    _r_strand = ((124 - 14) * _radians_to_angle, (124 + 14) * _radians_to_angle)
    _a_strand = (
        (-180) * _radians_to_angle,
        (-125) * _radians_to_angle,
        (145) * _radians_to_angle,
        (180) * _radians_to_angle,
    )
    _d2_strand = ((6.7 - 0.6), (6.7 + 0.6))
    _d3_strand = ((9.9 - 0.9), (9.9 + 0.9))
    _d4_strand = ((12.4 - 1.1), (12.4 + 1.1))

    # Filter all CA atoms in the relevant chain.

    d2i_coord = np.full((len(ca_coord), 2, 3), np.nan)
    d3i_coord = np.full((len(ca_coord), 2, 3), np.nan)
    d4i_coord = np.full((len(ca_coord), 2, 3), np.nan)
    ri_coord = np.full((len(ca_coord), 3, 3), np.nan)
    ai_coord = np.full((len(ca_coord), 4, 3), np.nan)

    # The distances and angles are not defined for the entire interval,
    # therefore the indices do not have the full range
    # Values that are not defined are NaN
    for i in range(1, len(ca_coord) - 1):
        d2i_coord[i] = (ca_coord[i - 1], ca_coord[i + 1])
    for i in range(1, len(ca_coord) - 2):
        d3i_coord[i] = (ca_coord[i - 1], ca_coord[i + 2])
    for i in range(1, len(ca_coord) - 3):
        d4i_coord[i] = (ca_coord[i - 1], ca_coord[i + 3])
    for i in range(1, len(ca_coord) - 1):
        ri_coord[i] = (ca_coord[i - 1], ca_coord[i], ca_coord[i + 1])
    for i in range(1, len(ca_coord) - 2):
        ai_coord[i] = (ca_coord[i - 1], ca_coord[i], ca_coord[i + 1], ca_coord[i + 2])

    d2i = distance(d2i_coord[:, 0], d2i_coord[:, 1])
    d3i = distance(d3i_coord[:, 0], d3i_coord[:, 1])
    d4i = distance(d4i_coord[:, 0], d4i_coord[:, 1])
    ri = angle(ri_coord[:, 0], ri_coord[:, 1], ri_coord[:, 2])
    ai = dihedral(ai_coord[:, 0], ai_coord[:, 1], ai_coord[:, 2], ai_coord[:, 3])

    sse = ["L"] * len(ca_coord)

    # Annotate helices
    # Find CA that meet criteria for potential helices
    is_pot_helix = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (
            d3i[i] >= _d3_helix[0]
            and d3i[i] <= _d3_helix[1]
            and d4i[i] >= _d4_helix[0]
            and d4i[i] <= _d4_helix[1]
        ) or (
            ri[i] >= _r_helix[0]
            and ri[i] <= _r_helix[1]
            and ai[i] >= _a_helix[0]
            and ai[i] <= _a_helix[1]
        ):
            is_pot_helix[i] = True
    # Real helices are 5 consecutive helix elements
    is_helix = np.zeros(len(sse), dtype=bool)
    counter = 0
    for i in range(len(sse)):
        if is_pot_helix[i]:
            counter += 1
        else:
            if counter >= 5:
                is_helix[i - counter : i] = True
            counter = 0
    # Extend the helices by one at each end if CA meets extension criteria
    i = 0
    while i < len(sse):
        if is_helix[i]:
            sse[i] = "H"
            if (d3i[i - 1] >= _d3_helix[0] and d3i[i - 1] <= _d3_helix[1]) or (
                ri[i - 1] >= _r_helix[0] and ri[i - 1] <= _r_helix[1]
            ):
                sse[i - 1] = "H"
            sse[i] = "H"
            if (d3i[i + 1] >= _d3_helix[0] and d3i[i + 1] <= _d3_helix[1]) or (
                ri[i + 1] >= _r_helix[0] and ri[i + 1] <= _r_helix[1]
            ):
                sse[i + 1] = "H"
        i += 1

    # Annotate sheets
    # Find CA that meet criteria for potential strands
    is_pot_strand = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (
            d2i[i] >= _d2_strand[0]
            and d2i[i] <= _d2_strand[1]
            and d3i[i] >= _d3_strand[0]
            and d3i[i] <= _d3_strand[1]
            and d4i[i] >= _d4_strand[0]
            and d4i[i] <= _d4_strand[1]
        ) or (
            ri[i] >= _r_strand[0]
            and ri[i] <= _r_strand[1]
            and (
                (ai[i] >= _a_strand[0] and ai[i] <= _a_strand[1])
                or (ai[i] >= _a_strand[2] and ai[i] <= _a_strand[3])
            )
        ):
            is_pot_strand[i] = True
    # Real strands are 5 consecutive strand elements,
    # or shorter fragments of at least 3 consecutive strand residues,
    # if they are in hydrogen bond proximity to 5 other residues
    is_strand = np.zeros(len(sse), dtype=bool)
    counter = 0
    contacts = 0
    for i in range(len(sse)):
        if is_pot_strand[i]:
            counter += 1
            coord = ca_coord[i]
            for strand_coord in ca_coord:
                dist = distance(coord, strand_coord)
                if dist >= 4.2 and dist <= 5.2:
                    contacts += 1
        else:
            if counter >= 4:
                is_strand[i - counter : i] = True
            elif counter == 3 and contacts >= 5:
                is_strand[i - counter : i] = True
            counter = 0
            contacts = 0
    # Extend the strands by one at each end if CA meets extension criteria
    i = 0
    while i < len(sse):
        if is_strand[i]:
            sse[i] = "E"
            if d3i[i - 1] >= _d3_strand[0] and d3i[i - 1] <= _d3_strand[1]:
                sse[i - 1] = "E"
            sse[i] = "E"
            if d3i[i + 1] >= _d3_strand[0] and d3i[i + 1] <= _d3_strand[1]:
                sse[i + 1] = "E"
        i += 1
    return sse


def make_ss_block_adj_from_structure(structure: protein.ProteinBase):
    secstruc_dict = extract_secstruc(structure)
    ss, idx = ss_to_tensor(secstruc_dict)
    block_adj = construct_block_adj_matrix(
        torch.FloatTensor(ss), torch.tensor(structure.atom_positions)
    ).float()

    ss_tens, _ = mask_ss(ss, idx, max_mask=0)
    ss_argmax = torch.argmax(ss_tens[:, :4], dim=1).float()
    return ss_argmax, block_adj
