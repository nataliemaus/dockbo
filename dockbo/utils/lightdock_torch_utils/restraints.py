import os 

def parse_restraints_file(restraints_file_name):
    """Parse a restraints file, returns a dictionary for receptor and ligand"""
    with open(restraints_file_name) as input_restraints:
        raw_restraints = [
            line.rstrip(os.linesep) for line in input_restraints.readlines()
        ]
        restraints = {
            "receptor": {"active": [], "passive": [], "blocked": []},
            "ligand": {"active": [], "passive": [], "blocked": []},
        }
        for restraint in raw_restraints:
            if restraint and restraint[0] in ["R", "L"]:
                try:
                    fields = restraint.split()
                    residue_identifier = fields[1].split(".")
                    # Only consider first character if many
                    chain_id = residue_identifier[0][0].upper()
                    # Only considering 3 chars if many
                    residue = residue_identifier[1][0:3].upper()
                    # Check for integer
                    try:
                        residue_number = int(residue_identifier[2])
                        residue_insertion = ""
                    except ValueError:
                        # Possible residue insertion
                        residue_number = int(residue_identifier[2][:-1])
                        residue_insertion = residue_identifier[2][-1].upper()
                    parsed_restraint = (
                        f"{chain_id}.{residue}.{residue_number}{residue_insertion}"
                    )
                    # Check type of restraint:
                    active = passive = blocked = False
                    try:
                        restraint_type = fields[2][0].upper()
                        passive = restraint_type == "P"
                        blocked = restraint_type == "B"
                        active = restraint_type == "A"
                    except (IndexError, AttributeError):
                        active = True

                    if fields[0] == "R":
                        if (
                            parsed_restraint not in restraints["receptor"]["active"]
                            and parsed_restraint
                            not in restraints["receptor"]["passive"]
                        ):
                            if active:
                                restraints["receptor"]["active"].append(
                                    parsed_restraint
                                )
                            elif passive:
                                restraints["receptor"]["passive"].append(
                                    parsed_restraint
                                )
                            elif blocked:
                                restraints["receptor"]["blocked"].append(
                                    parsed_restraint
                                )
                            else:
                                pass
                    else:
                        if (
                            parsed_restraint not in restraints["ligand"]["active"]
                            and parsed_restraint not in restraints["ligand"]["passive"]
                        ):
                            if active:
                                restraints["ligand"]["active"].append(parsed_restraint)
                            elif passive:
                                restraints["ligand"]["passive"].append(parsed_restraint)
                            elif blocked:
                                restraints["ligand"]["blocked"].append(parsed_restraint)
                            else:
                                pass
                except (AttributeError, IndexError):
                    pass 

        return restraints


def get_both_restraints(
    path_to_restraints_file,
    receptor,
    ligand,
):
    restraints = parse_restraints_file(path_to_restraints_file)
    # Calculate number of restraints in order to check them 
    num_rec_active = len(restraints["receptor"]["active"])
    num_rec_passive = len(restraints["receptor"]["passive"])
    num_rec_blocked = len(restraints["receptor"]["blocked"])
    num_lig_active = len(restraints["ligand"]["active"])
    num_lig_passive = len(restraints["ligand"]["passive"])

    # Complain if not a single restraint has been defined, but restraints are enabled
    if (
        not num_rec_active
        and not num_rec_passive
        and not num_rec_blocked
        and not num_lig_active
        and not num_lig_passive
    ):
        raise RuntimeError(
            "Restraints file specified, but not a single restraint found"
        )

    # Check if restraints correspond with real residues
    receptor_restraints = get_restraints(receptor, restraints["receptor"])
    # receptor_restraints = restraints["receptor"] 
    ligand_restraints = get_restraints(ligand, restraints["ligand"])
    # ligand_restraints = restraints["ligand"]

    return receptor_restraints, ligand_restraints 


def get_restraints(structure, restraints):
    """Check for each restraint in the format Chain.ResidueName.ResidueNumber in
    restraints if they exist in the given structure.
    """
    residues = {"active": [], "passive": [], "blocked": []}
    for restraint_type in ["active", "passive", "blocked"]:
        for restraint in restraints[restraint_type]:
            chain_id, residue_name, residue_number = restraint.split(".")
            if residue_number[-1].isalpha():
                residue_insertion = residue_number[-1]
                residue_number = residue_number[:-1]
            else:
                residue_insertion = ""
            residue = structure.get_residue(
                chain_id, residue_name, residue_number, residue_insertion
            )
            if not residue:
                raise RuntimeError(f"Restraint {restraint} not found in structure")
            residues[restraint_type].append(residue)
    return residues
