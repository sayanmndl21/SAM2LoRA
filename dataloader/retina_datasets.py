import random
from tqdm import tqdm
from typing import Any, List
from dataloader import (
    CHASEDBDataset,
    DRIVEDataset,
    FIVESDataset,
    HRFDataset,
    STAREDataset,
    DRISHTIGSDataset,
    G1020Dataset,
    GRAPEDataset,
    IDRIDDataset,
    ORIGADataset,
    PAPILADBDataset,
    REFUGE2Dataset
)
from dictionaries import VESSEL_DATASET_DICT, DISC_DATASET_DICT


def get_dataset(
    dataset_name: str = 'chasedb1',
    mode: str = 'train',
    seg_type: str = 'od',
    unified: bool = True,
    transform: Any = None,
    color_transform: Any = None,
    num_pos_points: int = 10,
    num_neg_points: int = 0,
    num_boxes: int = 1,
    region: str = 'general',
    random_state: int = 0,
) -> Any:
    dataset_type = mode
    if dataset_name.lower() in 'chasedb1':
        dataset = CHASEDBDataset(
            VESSEL_DATASET_DICT['CHASEDB1'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
    elif dataset_name.lower() in 'drive':
        dataset = DRIVEDataset(
            VESSEL_DATASET_DICT['DRIVE'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
    elif dataset_name.lower() in 'fives':
        dataset = FIVESDataset(
            VESSEL_DATASET_DICT['FIVES'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
    elif dataset_name.lower() in 'hrf':
        dataset = HRFDataset(
            VESSEL_DATASET_DICT['HRF'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
    elif dataset_name.lower() in 'stare':
        dataset = STAREDataset(
            VESSEL_DATASET_DICT['STARE'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
    elif dataset_name.lower() in 'drishtigs':
        dataset = DRISHTIGSDataset(
            DISC_DATASET_DICT['DRISHTIGS'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
            seg_type=seg_type,
        )
    elif dataset_name.lower() in 'g1020':
        dataset = G1020Dataset(
            DISC_DATASET_DICT['G1020'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
            seg_type=seg_type,
        )
    elif dataset_name.lower() in 'grape':
        dataset = GRAPEDataset(
            DISC_DATASET_DICT['GRAPE'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
            seg_type=seg_type,
        )
    elif dataset_name.lower() in 'idrid':
        dataset = IDRIDDataset(
            DISC_DATASET_DICT['IDRID'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
            seg_type=seg_type,
        )
    elif dataset_name.lower() in 'origa':
        dataset = ORIGADataset(
            DISC_DATASET_DICT['ORIGA'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
            seg_type=seg_type,
        )
    elif dataset_name.lower() in 'papiladb':
        dataset = PAPILADBDataset(
            DISC_DATASET_DICT['PAPILADB'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
            seg_type=seg_type,
        )
    elif dataset_name.lower() in 'refuge2':
        dataset = REFUGE2Dataset(
            DISC_DATASET_DICT['REFUGE2'],
            dataset_type=dataset_type,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
            seg_type=seg_type,
        )
    else:
        raise ValueError("Please dataset_name")
    return dataset


def generate_combined_batched_iterator(
    datasets: List[Any],
    batch_size: int = 8
) -> Any:
    # Generate iterators for each dataset
    dataset_iterators = [ds.get_batches(batch_size=batch_size, dataset_type='train') for ds in datasets]
    
    total_batches = sum(len(ds) for ds in datasets) // batch_size
    combined_batches = []
    with tqdm(total=total_batches, desc=f"Combining {total_batches} Batches") as pbar:
        for iterator in dataset_iterators:
            for batch in iterator:
                combined_batches.append(batch)
                pbar.update(1)
    
    random.shuffle(combined_batches)
    for images, masks, pos_points, neg_points, boxes, num_masks in combined_batches:
        yield images, masks, pos_points, neg_points, boxes, num_masks


def get_batched_dataset(
    dataset_type: str = 'vessel',
    batch_size: int = 16,
    mode: str = 'train',
    seg_type: str = 'od',
    unified: bool = True,
    transform: Any = None,
    color_transform: Any = None,
    num_pos_points: int = 10,
    num_neg_points: int = 0,
    num_boxes: int = 1,
    region: str = 'general',
    random_state: int = 0
) -> Any:
    if dataset_type not in ['vessel', 'optic_disc']:
        raise ValueError('dataset_type should be vessel or optic_disc')
    
    if dataset_type == 'vessel':
        dataset1 = get_dataset(
            dataset_name='chasedb1',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset2 = get_dataset(
            dataset_name='drive',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset3 = get_dataset(
            dataset_name='fives',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset4 = get_dataset(
            dataset_name='hrf',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset5 = get_dataset(
            dataset_name='stare',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset_list = [dataset1, dataset2, dataset3, dataset4, dataset5]
    elif dataset_type == 'optic_disc':
        dataset1 = get_dataset(
            dataset_name='drishtigs',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            seg_type=seg_type,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset2 = get_dataset(
            dataset_name='g1020',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            seg_type=seg_type,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset3 = get_dataset(
            dataset_name='grape',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            seg_type=seg_type,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset4 = get_dataset(
            dataset_name='origa',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            seg_type=seg_type,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset5 = get_dataset(
            dataset_name='papiladb',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            seg_type=seg_type,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset6 = get_dataset(
            dataset_name='refuge2',
            mode=mode,
            unified=unified,
            transform=transform,
            color_transform=color_transform,
            seg_type=seg_type,
            num_pos_points=num_pos_points,
            num_neg_points=num_neg_points,
            num_boxes=num_boxes,
            region=region,
            random_state=random_state,
        )
        dataset_list = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6]
    return generate_combined_batched_iterator(dataset_list, batch_size=batch_size)


def combine_batched_iterators(batch_iterators: List[Any]) -> Any:
    combined_batches = []
    for batch_iterator in batch_iterators:
        for batch in batch_iterator:
            combined_batches.append(batch)
    random.shuffle(combined_batches)
    for batch in combined_batches:
        yield batch