import os
import subprocess
import uuid
from abc import ABC
from typing import List
import Bio.PDB
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class StructurePredictor(ABC):
    def __init__(self, out_dir: str, name: str):
        """
        Because all StructurePredictors create .pdb files, they each need a directory to store them at a minimum
        """
        self.out_dir = out_dir
        self.name = name
        pass
    def fold(self, sequences: list, store: bool) -> List[Bio.PDB.Structure.Structure]:
        """
        Must input list of sequences and output list of biopython structure objects based on the model's predicted
        structure. If store is false, delete temporary structure files. Otherwise, leave them in self.out_dir
        sequences: list of strings for which structures will be generated
        store: boolean dictating whether pdb files will be deleted
        returns: list of biopython structure objects of the predicted structures in the same order as sequences
            ** Editied by nmaus to return a list of filenames for pdbs instead when store == True 
        """
        pass
    def get_file_name(self, seq: str, *args: str) -> str:
        """
        Returns a standardized name for structure files with hashed sequences
        seq: sequence being folded
        *args: [Optional] additional strings that are appended to the end of the name for more clarification
        """
        arg_list = "".join([f"_{a}" for a in args])
        return f"{hash(seq)}_{self.name}{arg_list}.pdb"


class NanoNet(StructurePredictor):
    """
    A very fast nanobody-only structure predictor that uses a simple architecture to take in sequences as padded one-hot
    vectors and output positions for alpha carbons. Full structure is then generated from these alpha carbon predictions
    https://www.biorxiv.org/content/10.1101/2021.08.03.454917v1.full.pdf
    """
    def __init__(self, out_dir: str, name='nanonet', nanonet_path='~/NanoNet/NanoNet.py', side_chains=False):
        """
        out_dir: the directory where .pdb files will be stored
        name: the name of the model
        side_chains: whether or not to solve for side chain atoms using modeller (by default just does backbone + Cb)
        """
        super().__init__(out_dir, name)
        self.nanonet_path = nanonet_path
        self.side_chains = side_chains
        if self.side_chains:
            raise NotImplementedError("Haven't implemented the refine feature yet")

    def fold(self, sequences: list, store: bool) -> List[Bio.PDB.Structure.Structure]:
        temp_out_dir = self.out_dir + '/' + str(uuid.uuid1())
        os.mkdir(temp_out_dir)
        parser = Bio.PDB.PDBParser()
        structures = []
        # save fasta of sequences
        seq_records = [
            SeqRecord(
                Seq(s),
                id=str(i)
            ) for i, s in enumerate(sequences)
        ]
        fasta_file = os.path.join(temp_out_dir, "temp.fasta")
        SeqIO.write(seq_records, fasta_file, "fasta")
        # run model from shell command
        cmd = f"python {self.nanonet_path} {fasta_file} -o {temp_out_dir}{' -m' if self.side_chains else ''}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
        # cleanup
        os.remove(fasta_file)  # always remove fasta file
        filenames = [""]*len(sequences)

        # for now the multiple seqs thing is broken... 
        assert len(sequences) == 1
        for f in os.listdir(temp_out_dir):
            if f[-3:] != 'pdb':  # sometimes there are hidden ipynb checkpoint files you need to skip
                continue
            if len(f.split('_')[0]) < 15:  # one of the new structures
                structures.append(parser.get_structure('', os.path.join(temp_out_dir, f)))  # parse structure
                if store:
                    seq_n = 0
                    # seq_n = f.split('_')[0]
                    # print("seq_n 1", seq_n)
                    # seq_n = seq_n[0:int(len(seq_n)//2)]
                    # # "nanot_0"
                    # print("seq_n 2", seq_n)
                    # seq_n = int(seq_n)
                    # print("seq_n 3", seq_n)
                    new_f = f"{str(uuid.uuid4())}_{self.name}_{'side_chains' if self.side_chains else 'bb'}.pdb"
                    # print("new_f", new_f)
                    os.rename(
                        os.path.join(temp_out_dir, f),
                        os.path.join(temp_out_dir, new_f)
                    )
                    filenames[seq_n] = temp_out_dir + '/' + new_f
                else:
                    os.remove(os.path.join(temp_out_dir, f))
        if store:
            return filenames, temp_out_dir
        else:
            return structures


if __name__ == "__main__":
    ''' Example below runs nanonet to create 
        pdb files for two example protein sequences
    '''
    example_seq1 = "QLQADDKGLAQDGGSLRLSCAYSGDTVNDYAMAWFRQAPGKGREFVAAIRARGGGTEYLDSVKGPDDISRDNGENTAYLQMDNLQPDDKPEYFCALAMGGYAYRAFERYSVRGQGTQVTVS"
    example_seq2 = "QVQLSGLSCLAQAYSGDTVNDYAMAWFRQATLRQLQADDKGDGTLRGS"
    protein_sequences = [example_seq1,example_seq2]
    save_nanonet_pdb_files_directory = 'nanonet_generated_pdb_files'
    if not os.path.exists(save_nanonet_pdb_files_directory):
        os.mkdir(save_nanonet_pdb_files_directory)
    nanonet = NanoNet(out_dir=save_nanonet_pdb_files_directory)
    paths_to_new_pdb_files, _ = nanonet.fold(protein_sequences, store=True)
    # print(f'New pdb file for example seq 1 is stored at : {paths_to_new_pdb_files[0]}')
    # print(f'New pdb file for example seq 2 is stored at : {paths_to_new_pdb_files[1]}')
