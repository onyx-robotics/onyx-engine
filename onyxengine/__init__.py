# onyxengine/__init__.py
import os

# API Constants
SERVER_URL = "https://api.onyx-robotics.com"
#SERVER_URL = "http://localhost:8000"
ONYX_API_KEY = os.environ.get('ONYX_API_KEY')
if ONYX_API_KEY is None:
    print('Warning ONYX_API_KEY environment variable not found.')
ONYX_PATH = './onyx'
DATASETS_PATH = os.path.join(ONYX_PATH, 'datasets')
MODELS_PATH = os.path.join(ONYX_PATH, 'models')

from .api import *