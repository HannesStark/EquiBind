from torch.utils.data import Dataset
from commons.process_mols import get_geometry_graph, get_lig_graph_revised, get_rdkit_coords
from dgl import batch
from rdkit.Chem import SDMolSupplier, SanitizeMol, SanitizeFlags, PropertyMol, SmilesMolSupplier, AddHs


class Ligands(Dataset):
    def __init__(self, ligpath, rec_graph, args, lazy = None, slice = None, skips = None, ext = None, addH = None, rdkit_seed = None):
        self.ligpath = ligpath
        self.rec_graph = rec_graph
        self.args = args
        self.dp = args.dataset_params
        self.use_rdkit_coords = args.use_rdkit_coords
        self.device = args.device
        self.rdkit_seed = rdkit_seed
        
        ##Default argument handling
        self.skips = skips if skips is not None else set()

        extensions_requiring_conformer_generation = ["smi"]
        extensions_defaulting_to_lazy = ["smi"]

        if ext is None:
            try:
                ext = ligpath.split(".")[-1]
            except (AttributeError, KeyError):
                ext = "sdf"
        

        if lazy is None:
            if ext in extensions_defaulting_to_lazy:
                self.lazy = True
            else:
                self.lazy = False
        else:
            self.lazy = lazy

        if addH is None:
            if ext == "smi":
                addH = True
            else:
                addH = False
        self.addH = addH
        
        self.generate_conformer = ext in extensions_requiring_conformer_generation

        suppliers = {"sdf": SDMolSupplier, "smi": SmilesMolSupplier}
        supp_kwargs = {"sdf": dict(sanitize = False, removeHs =  False),
                        "smi": dict(sanitize = False)}
        self.supplier = suppliers[ext](ligpath, **supp_kwargs[ext])

        if slice is None:
            self.slice = 0, len(self.supplier)
        else:
            slice = (slice[0] if slice[0] >= 0 else len(self.supplier)+slice[0], slice[1] if slice[1] >= 0 else len(self.supplier)+slice[1])
            self.slice = tuple(slice)

        self.failed_ligs = []
        self.true_idx = []

        if not self.lazy:
            self.ligs = []
            for i in range(*self.slice):
                if i in self.skips:
                    continue
                lig = self.supplier[i]
                lig, name = self._process(lig)
                if lig is not None:
                    self.ligs.append(PropertyMol.PropertyMol(lig))
                    self.true_idx.append(i)
                else:
                    self.failed_ligs.append((i, name))

        if self.lazy:
            self._len = self.slice[1]-self.slice[0]
        else:
            self._len = len(self.ligs)

    def _process(self, lig):
        if lig is None:
            return None, None
        if self.addH:
            lig = AddHs(lig)
        if self.generate_conformer:
            get_rdkit_coords(lig, self.rdkit_seed)
        sanitize_succeded = (SanitizeMol(lig, catchErrors = True) is SanitizeFlags.SANITIZE_NONE)
        if sanitize_succeded:
            return lig, lig.GetProp("_Name")
        else:
            return None, lig.GetProp("_Name")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.lazy:
            if idx < 0:
                nonneg_idx = self._len + idx
            else:
                nonneg_idx = idx

            if nonneg_idx >= self._len or nonneg_idx < 0:
                raise IndexError(f"Index {idx} out of range for Ligands dataset with length {len(self)}")
            
            
            true_index = nonneg_idx + self.slice[0]
            if true_index in self.skips:
                return true_index, "Skipped"
            lig = self.supplier[true_index]
            lig, name = self._process(lig)
            if lig is not None:
                lig = PropertyMol.PropertyMol(lig)
            else:
                self.failed_ligs.append((true_index, name))
                return true_index, name
        elif not self.lazy:
            lig = self.ligs[idx]
            true_index = self.true_idx[idx]

        
        try:
            lig_graph = get_lig_graph_revised(lig, lig.GetProp('_Name'), max_neighbors=self.dp['lig_max_neighbors'],
                                            use_rdkit_coords=self.use_rdkit_coords, radius=self.dp['lig_graph_radius'])
        except AssertionError:
            self.failed_ligs.append((true_index, lig.GetProp("_Name")))
            return true_index, lig.GetProp("_Name")
        
        geometry_graph = get_geometry_graph(lig) if self.dp['geometry_regularization'] else None

        lig_graph.ndata["new_x"] = lig_graph.ndata["x"]
        return lig, lig_graph.ndata["new_x"], lig_graph, self.rec_graph, geometry_graph, true_index
    
    @staticmethod
    def collate(_batch):
        sample_succeeded = lambda sample: not isinstance(sample[0], int)
        sample_failed = lambda sample: isinstance(sample[0], int)
        clean_batch = tuple(filter(sample_succeeded, _batch))
        failed_in_batch = tuple(filter(sample_failed, _batch))
        if len(clean_batch) == 0:
            return None, None, None, None, None, None, failed_in_batch
        ligs, lig_coords, lig_graphs, rec_graphs, geometry_graphs, true_indices = map(list, zip(*clean_batch))
        output = (
            ligs,
            lig_coords,
            batch(lig_graphs),
            batch(rec_graphs),
            batch(geometry_graphs) if geometry_graphs[0] is not None else None,
            true_indices,
            failed_in_batch
        )
        return output
