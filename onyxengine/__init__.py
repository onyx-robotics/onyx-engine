# onyxengine/__init__.py
import os

# API Constants
SERVER = "api.onyx-robotics.com"
#SERVER = "localhost:8000"
SERVER_URL = f"https://{SERVER}" if SERVER != "localhost:8000" else f"http://{SERVER}"
WSS_URL = f"wss://{SERVER}/ws" if SERVER != "localhost:8000" else f"ws://{SERVER}/ws"
ONYX_API_KEY = os.environ.get('ONYX_API_KEY')
if ONYX_API_KEY is None:
    print('Warning ONYX_API_KEY environment variable not found.')
ONYX_PATH = './onyx'
DATASETS_PATH = os.path.join(ONYX_PATH, 'datasets')
MODELS_PATH = os.path.join(ONYX_PATH, 'models')

from .api import *