from pathlib import Path
import os

#  PROJ_DIR = Path(__file__).parent.resolve().parent

path = Path()
PROJ_DIR = path.resolve()

DATA_DIR = PROJ_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
TRANSFORMED_DATA_DIR = DATA_DIR / 'transformed'
DATA_CACHE_DIR = DATA_DIR / 'cache'

MODELS_DIR = PROJ_DIR / 'models'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)

if not Path(DATA_CACHE_DIR).exists():
    os.mkdir(DATA_CACHE_DIR)