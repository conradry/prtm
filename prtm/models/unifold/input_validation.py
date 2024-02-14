from typing import Dict, Sequence, Tuple

from prtm.models.unifold.data import residue_constants


def clean_and_validate_sequence(
    input_sequence: str, min_length: int, max_length: int
) -> str:
    """Checks that the input sequence is ok and returns a clean version of it."""
    # Remove all whitespaces, tabs and end lines; upper-case.
    clean_sequence = input_sequence.translate(str.maketrans("", "", " \n\t")).upper()
    aatypes = set(residue_constants.restypes)  # 20 standard aatypes.
    if not set(clean_sequence).issubset(aatypes):
        raise ValueError(
            f"Input sequence contains non-amino acid letters: "
            f"{set(clean_sequence) - aatypes}. AlphaFold only supports 20 standard "
            "amino acids as inputs."
        )
    if len(clean_sequence) < min_length:
        raise ValueError(
            f"Input sequence is too short: {len(clean_sequence)} amino acids, "
            f"while the minimum is {min_length}"
        )
    if len(clean_sequence) > max_length:
        raise ValueError(
            f"Input sequence is too long: {len(clean_sequence)} amino acids, while "
            f"the maximum is {max_length}. You may be able to run it with the full "
            f"Uni-Fold system depending on your resources (system memory, "
            f"GPU memory)."
        )
    return clean_sequence


def validate_input(
    input_sequences: Dict[str, str],
    symmetry_group: str,
    min_length: int,
    max_length: int,
    max_multimer_length: int,
) -> Tuple[Sequence[str], bool]:
    """Validates and cleans input sequences and determines which model to use."""
    sequences = {}
    for chain_id, input_sequence in input_sequences.items():
        if input_sequence.strip():
            input_sequence = clean_and_validate_sequence(
                input_sequence=input_sequence,
                min_length=min_length,
                max_length=max_length,
            )
            sequences[chain_id] = input_sequence

    if symmetry_group != "C1" and symmetry_group is not None:
        if symmetry_group.startswith("C") and symmetry_group[1:].isnumeric():
            print(
                f"Using UF-Symmetry with group {symmetry_group}. If you do not "
                f"want to use UF-Symmetry, please use `C1` and copy the AU "
                f"sequences to the count in the assembly."
            )
            is_multimer = len(sequences) > 1
            return sequences, is_multimer, symmetry_group
        else:
            raise ValueError(
                f"UF-Symmetry does not support symmetry group "
                f"{symmetry_group} currently. Cyclic groups (Cx) are "
                f"supported only."
            )

    elif len(sequences) == 1:
        return sequences, False, None

    elif len(sequences) > 1:
        total_multimer_length = sum([len(seq) for seq in sequences.values()])
        if total_multimer_length > max_multimer_length:
            raise ValueError(
                f"The total length of multimer sequences is too long: "
                f"{total_multimer_length}, while the maximum is "
                f"{max_multimer_length}. Please use the full AlphaFold "
                f"system for long multimers."
            )
        return sequences, True, None

    else:
        raise ValueError(
            "No input amino acid sequence provided, please provide at "
            "least one sequence."
        )
