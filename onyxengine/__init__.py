# onyxengine/__init__.py
import os

# API Constants
DEV_MODE = False
SERVER = "api.onyx-robotics.com"
DEV_SERVER = "localhost:7000"
SERVER_URL = f"https://{SERVER}" if not DEV_MODE else f"http://{DEV_SERVER}"
WSS_URL = f"wss://{SERVER}/ws" if not DEV_MODE else f"ws://{DEV_SERVER}/ws"
ONYX_API_KEY = os.environ.get('ONYX_API_KEY')
if ONYX_API_KEY is None:
    print('Warning ONYX_API_KEY environment variable not found.')
ONYX_PATH = './onyx'
DATASETS_PATH = os.path.join(ONYX_PATH, 'datasets')
MODELS_PATH = os.path.join(ONYX_PATH, 'models')

from .api import *