from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from torch_geometric.nn import radius_graph
from e3nn.o3 import Irreps, spherical_harmonics


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
           'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']

thermo_targets = ['U', 'U0', 'H', 'G']


class TargetGetter(object):
    """ Gets relevant target """

    def __init__(self, target):
        self.target = target
        self.target_idx = targets.index(target)

    def __call__(self, data):
        # Specify target.
        data.y = data.y[0, self.target_idx]
        return data


class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper. """

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root, target, radius, partition, lmax_attr, feature_type="one_hot"):
        assert feature_type in ["one_hot", "cormorant", "gilmer"], "Please use valid features"
        assert target in targets
        assert partition in ["train", "valid", "test"]
        self.root = osp.abspath(osp.join(root, "qm9"))
        self.target = target
        self.radius = radius
        self.partition = partition
        self.feature_type = feature_type
        self.lmax_attr = lmax_attr
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        transform = TargetGetter(self.target)

        super().__init__(self.root, transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def calc_stats(self):
        ys = np.array([data.y.item() for data in self])
        mean = np.mean(ys)
        mad = np.mean(np.abs(ys - mean))
        return mean, mad

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @ property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            print("Please install rdkit")
            return

    @ property
    def processed_file_names(self) -> str:
        return ["_".join([self.partition, "r="+str(np.round(self.radius, 2)),
                          self.feature_type, "l="+str(self.lmax_attr)]) + '.pt']

    def download(self):
        print("i'm downloading", self.raw_dir, self.raw_url)
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            print("Please install rdkit")
            return

        print("Processing", self.partition, "with radius=" + str(np.round(self.radius, 2)) +
              ",", "l_attr=" + str(self.lmax_attr), "and", self.feature_type, "features.")
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)
        data_list = []

        # Create splits identical to Cormorant
        Nmols = len(suppl) - len(skip)
        Ntrain = 100000
        Ntest = int(0.1*Nmols)
        Nvalid = Nmols - (Ntrain + Ntest)

        np.random.seed(0)
        data_perm = np.random.permutation(Nmols)
        train, valid, test = np.split(data_perm, [Ntrain, Ntrain+Nvalid])
        indices = {"train": train, "valid": valid, "test": test}

        # Add a very ugly second index to align with Cormorant splits.
        j = 0
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue
            if j not in indices[self.partition]:
                j += 1
                continue
            j += 1

            N = mol.GetNumAtoms()

            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            edge_index = radius_graph(pos, r=self.radius, loop=False)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            if self.feature_type == "one_hot":
                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            elif self.feature_type == "cormorant":
                one_hot = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
                x = get_cormorant_features(one_hot, z, 2, z.max())
            elif self.feature_type == "gilmer":
                row, col = edge_index
                hs = (z == 1).to(torch.float)
                num_hs = scatter(hs[row], col, dim_size=N).tolist()

                x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
                x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                                  dtype=torch.float).t().contiguous()
                x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')

            edge_attr, node_attr, edge_dist = self.get_O3_attr(edge_index, pos, self.attr_irreps)

            data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                        node_attr=node_attr, additional_message_features=edge_dist, y=y, name=name, index=i)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_O3_attr(self, edge_index, pos, attr_irreps):
        """ Creates spherical harmonic edge attributes and node attributes for the SEGNN """
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]  # pos_j - pos_i (note in edge_index stores tuples like (j,i))
        edge_dist = rel_pos.pow(2).sum(-1, keepdims=True)
        edge_attr = spherical_harmonics(attr_irreps, rel_pos, normalize=True,
                                        normalization='component')  # Unnormalised for now
        node_attr = scatter(edge_attr, edge_index[1], dim=0, reduce="mean")
        return edge_attr, node_attr, edge_dist


def get_cormorant_features(one_hot, charges, charge_power, charge_scale):
    """ Create input features as described in section 7.3 of https://arxiv.org/pdf/1906.04015.pdf """
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


if __name__ == "__main__":
    dataset = QM9("datasets", "alpha", 2.0, "train", feature_type="one_hot")
    print("length", len(dataset))
    ys = np.array([data.y.item() for data in dataset])
    mean, mad = dataset.calc_stats()

    for item in dataset:
        print(item.edge_index)
        break

    print("mean", mean, "mad", mad)
    import matplotlib.pyplot as plt

    plt.subplot(121)
    plt.title(dataset.target)
    plt.hist(ys)
    plt.subplot(122)
    plt.title(dataset.target + " standardised")
    plt.hist((ys - mean)/mad)
    plt.show()
