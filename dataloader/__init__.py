from dataloader.chasedb import CHASEDBDataset
from dataloader.drive import DRIVEDataset
from dataloader.fives import FIVESDataset
from dataloader.hrf import HRFDataset
from dataloader.stare import STAREDataset
from dataloader.drishtigs import DRISHTIGSDataset
from dataloader.g1020 import G1020Dataset
from dataloader.grape import GRAPEDataset
from dataloader.idrid import IDRIDDataset
from dataloader.origa import ORIGADataset
from dataloader.papiladb import PAPILADBDataset
from dataloader.refuge2 import REFUGE2Dataset
from dataloader.retina_datasets import get_dataset 

__all__ = [
    "get_dataset",
"CHASEDBDataset",
"DRIVEDataset",
"FIVESDataset",
"HRFDataset",
"STAREDataset",
"DRISHTIGSDataset",
"G1020Dataset",
"GRAPEDataset",
"IDRIDDataset",
"ORIGADataset",
"PAPILADBDataset",
"REFUGE2Dataset"
]