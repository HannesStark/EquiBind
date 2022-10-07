from torch.utils.data import Dataset
from commons.process_mols import get_geometry_graph, get_lig_graph_revised, get_rdkit_coords
from dgl import batch
from rdkit.Chem import SDMolSupplier, SanitizeMol, SanitizeFlags, PropertyMol, SmilesMolSupplier, AddHs, MultithreadedSmilesMolSupplier, MultithreadedSDMolSupplier


class Ligands(Dataset):
    def __init__(self, ligpath, rec_graph, args, skips = None, ext = None, addH = None, rdkit_seed = None, lig_load_workers = 0):
        self.ligpath = ligpath
        self.rec_graph = rec_graph
        self.args = args
        self.dp = args.dataset_params
        self.use_rdkit_coords = args.use_rdkit_coords
        self.device = args.device
        self.rdkit_seed = rdkit_seed
        
        ##Default argument handling
        self.skips = skips

        extensions_requiring_conformer_generation = ["smi"]

        if ext is None:
            try:
                ext = ligpath.split(".")[-1]
            except (AttributeError, KeyError):
                ext = "sdf"

        if addH is None:
            if ext == "smi":
                addH = True
            else:
                addH = False
        self.addH = addH
        
        self.generate_conformer = ext in extensions_requiring_conformer_generation

        if lig_load_workers > 0:
            suppliers = {"sdf": MultithreadedSDMolSupplier, "smi": MultithreadedSmilesMolSupplier}
            supp_kwargs = {"sdf": dict(sanitize = False, removeHs =  False, numWriterThreads = lig_load_workers),
                            "smi": dict(sanitize = False, titleLine = False, numWriterThreads = lig_load_workers)}
            self.supplier = suppliers[ext](ligpath, **supp_kwargs[ext])
            print("start loading ligs")
            self.ligs = [(lig, self.supplier.GetLastRecordId()) for lig in self.supplier]
            self.ligs = sorted(self.ligs, key = lambda tup: tup[1])
            self.ligs = list(zip(*self.ligs))[0][:-1]
            print("finish loading ligs")
        else:
            suppliers = {"sdf": SDMolSupplier, "smi": SmilesMolSupplier}
            supp_kwargs = {"sdf": dict(sanitize = False, removeHs =  False),
                            "smi": dict(sanitize = False, titleLine = False)}
            self.supplier = suppliers[ext](ligpath, **supp_kwargs[ext])
            self.ligs = [lig for lig in self.supplier]
        
        self._len = len(self.ligs)

    
    def _process(self, lig):
        if lig is None:
            return None, None
        if self.addH:
            lig = AddHs(lig)
        if self.generate_conformer:
            try:
                get_rdkit_coords(lig, self.rdkit_seed)
            except ValueError:
                return None, lig.GetProp("_Name")
        sanitize_succeded = (SanitizeMol(lig, catchErrors = True) is SanitizeFlags.SANITIZE_NONE)
        if sanitize_succeded:
            return lig, lig.GetProp("_Name")
        else:
            return None, lig.GetProp("_Name")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        lig = self.ligs[idx]

        if self.skips is not None and idx in self.skips:
            return idx, "Skipped"
        
        lig, name = self._process(lig)
        if lig is None:
            return idx, name
        else:
            lig = PropertyMol.PropertyMol(lig)

        try:
            lig_graph = get_lig_graph_revised(lig, lig.GetProp('_Name'), max_neighbors=self.dp['lig_max_neighbors'],
                                            use_rdkit_coords=self.use_rdkit_coords, radius=self.dp['lig_graph_radius'])
        except AssertionError:
            return idx, lig.GetProp("_Name")
        
        geometry_graph = get_geometry_graph(lig) if self.dp['geometry_regularization'] else None

        lig_graph.ndata["new_x"] = lig_graph.ndata["x"]
        return lig, lig_graph.ndata["new_x"], lig_graph, self.rec_graph, geometry_graph, idx
    
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
