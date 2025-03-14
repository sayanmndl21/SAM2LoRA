import os

# Set basepath to the current directory
basepath =  os.getcwd()

VESSEL_DATASET_DICT = {
    'CHASEDB1': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'CHASEDB1', 'CHASEDB1'),
    'DRIVE': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'DRIVE'),
    'FIVES': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'FIVES'),
    'HRF': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'HRF'),
    'STARE': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'STARE')
}

DISC_DATASET_DICT = {
    'DRISHTIGS': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'DRISHTI-GS'),
    'G1020': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'G1020'),
    'GRAPE': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'GRAPE'),
    'IDRID': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'IDRID', 'Segmentation'),
    'ORIGA': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'ORIGA'),
    'PAPILADB': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'PapilaDB'),
    'REFUGE2': os.path.join(basepath, 'datasets', 'eye_dataset', 'Fundus', 'REFUGE2')
}

CKPT_EXT = '.ckpt'