import os
import requests
from typing import List
import torch
import pandas as pd
from tqdm import tqdm
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from onyxengine.data import OnyxDataset, OnyxDatasetConfig
from onyxengine.modeling import model_from_config_json
from onyxengine.modeling.models import *

# API Constants
SERVER_URL = "https://api.onyx-robotics.com"
#SERVER_URL = "http://localhost:8000"
ONYX_API_KEY = os.environ.get('ONYX_API_KEY')
if ONYX_API_KEY is None:
    print('Warning ONYX_API_KEY environment variable not found.')
ONYX_PATH = './onyx'
DATASETS_PATH = os.path.join(ONYX_PATH, 'datasets')
MODELS_PATH = os.path.join(ONYX_PATH, 'models')

def upload_object(filename, object_type):
    # Get secure upload URL from the cloud
    try:
        response = requests.post(
            SERVER_URL + "/generate_upload_url",
            headers={"x-api-key": ONYX_API_KEY},
            json={"object_filename": filename, "object_type": object_type},
        )
        response.raise_for_status()
        response = response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while connecting with onyx server: {e}")
        print(f"Response content: {response.content.decode('utf-8')}")
        return

    # Upload the object using the secure URL
    local_copy_path = os.path.join(ONYX_PATH, object_type + 's', filename)
    file_size = os.path.getsize(local_copy_path)
    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar:
        with open(local_copy_path, "rb") as file:
            fields = response["fields"]
            fields["file"] = (filename, file)
            e = MultipartEncoder(fields=fields)
            m = MultipartEncoderMonitor(e, lambda monitor: progress_bar.update(monitor.bytes_read - progress_bar.n))
            headers = {"Content-Type": m.content_type}
            try:
                response = requests.post(response['url'], data=m, headers=headers)
                response.raise_for_status()
                print(f'{filename} uploaded successfully.')
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error occurred while uploading {filename} to the cloud: {e}")
                print(f"Response content: {response.content.decode('utf-8')}")

def download_object(filename, object_type):
    # Get secure download URL from the cloud
    try:
        response = requests.post(
            SERVER_URL + "/generate_download_url",
            headers={"x-api-key": ONYX_API_KEY},
            json={"object_filename": filename, "object_type": object_type},
        )
        response.raise_for_status()
        download_url = response.json()["download_url"]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while connecting with onyx server: {e}")
        print(f"Response content: {response.content.decode('utf-8')}")
        return

    # Download the object using the secure URL
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while downloading {filename} from the cloud: {e}")
        print(f"Response content: {response.content.decode('utf-8')}")
        return

    # Write the object to local storage
    block_size = 1024
    total_size = int(response.headers.get("content-length", 0))
    local_copy_path = os.path.join(ONYX_PATH, object_type + 's', filename)
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(local_copy_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    print(f'{filename} downloaded successfully.')

def set_object_metadata(object_id, object_type, object_config, object_source_ids=None):
    if object_source_ids is None:
        object_source_ids = []
    
    # Request to set metadata for the object in onyx engine
    try:
        response = requests.post(
            SERVER_URL + "/set_object_metadata",
            headers={"x-api-key": ONYX_API_KEY},
            json={
                "object_id": object_id,
                "object_type": object_type,
                "object_config": object_config,
                "object_source_ids": object_source_ids,
            },
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while connecting with onyx server: {e}")
        print(f"Response content: {response.content.decode('utf-8')}")

def get_object_metadata(object_id):
    # Get metadata for the object in onyx engine
    try:
        response = requests.post(
            SERVER_URL + "/get_object_metadata",
            headers={"x-api-key": ONYX_API_KEY},
            json={"object_id": object_id},
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while connecting with onyx server: {e}")
        print(f"Response content: {response.content.decode('utf-8')}")
    return response.json()

def save_dataset(object_id, dataset: OnyxDataset, source_dataset_ids: List[str] = None):
    # Save a local copy of the dataset and its config
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    dataset_filename = object_id + '.csv'
    config_filename = object_id + '.json'
    config_json = dataset.config.model_dump_json(indent=2)
    dataset.dataframe.to_csv(os.path.join(DATASETS_PATH, dataset_filename), index=False)
    with open(os.path.join(DATASETS_PATH, config_filename), 'w') as f:
        f.write(config_json)
    
    # Upload the dataset and config to the cloud
    upload_object(config_filename, 'dataset')
    upload_object(dataset_filename, 'dataset')
    set_object_metadata(object_id, 'dataset', config_json, source_dataset_ids)

def load_dataset(dataset_id):
    # Get dataset and config filenames
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    dataset_filename = dataset_id + '.csv'
    config_filename = dataset_id + '.json'
    dataset_path = os.path.join(DATASETS_PATH, dataset_filename)
    config_path = os.path.join(DATASETS_PATH, config_filename)
    
    # Download the dataset from the cloud if it doesn't exist locally
    if not os.path.exists(dataset_path):
        download_object(dataset_filename, 'dataset')
        download_object(config_filename, 'dataset')
            
    # Load the dataset and config from local storage
    dataset_dataframe = pd.read_csv(dataset_path)
    with open(config_path, 'r') as f:
        config_json = f.read()
    dataset_config = OnyxDatasetConfig.model_validate_json(config_json)
    dataset = OnyxDataset(dataframe=dataset_dataframe)
    dataset.from_config(dataset_config)
    
    return dataset

def save_model(object_id, model, model_config, source_dataset_ids: List[str]):    
    # Save model to local storage
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    model_filename = object_id + '.pt'
    config_filename = object_id + '.json'
    config_json = model_config.model_dump_json(indent=2)
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, model_filename))
    with open(os.path.join(MODELS_PATH, config_filename), 'w') as f:
        f.write(config_json)
    
    # Upload the dataset and config to the cloud
    upload_object(config_filename, 'model')
    upload_object(model_filename, 'model')
    set_object_metadata(object_id, 'model', config_json, source_dataset_ids)

def load_model(model_id):
    # Get model and config filenames
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    model_filename = model_id + '.pt'
    config_filename = model_id + '.json'
    model_path = os.path.join(MODELS_PATH, model_filename)
    config_path = os.path.join(MODELS_PATH, config_filename)
    
    # Download the model from the cloud if it doesn't exist locally
    if not os.path.exists(model_path):
        download_object(model_filename, 'model')
        download_object(config_filename, 'model')
            
    # Load the model using config and state_dict from local storage
    with open(config_path, 'r') as f:
        config_json = f.read()
    model = model_from_config_json(config_json)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def train_model(dataset_id, model_id, model_config, training_config):
    # Prepare to make a request to the onyx server
    model_config_json = model_config.model_dump_json(indent=2)
    training_config_json = training_config.model_dump_json(indent=2)

    # Request the onyx server to train the model
    try:
        response = requests.post(
            SERVER_URL + "/train_model",
            headers={"x-api-key": ONYX_API_KEY},
            json={
                "dataset_id": dataset_id,
                "onyx_model_id": model_id,
                "onyx_model_config": model_config_json,
                "training_config": training_config_json,
            },
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while connecting with onyx server: {e}")
        print(f"Response content: {response.content.decode('utf-8')}")