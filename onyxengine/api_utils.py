import os
import json
import requests
from typing import List, Optional
from pydantic import BaseModel
from tqdm import tqdm
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from onyxengine import SERVER_URL, WSS_URL, ONYX_PATH
from onyxengine.modeling import TrainingConfig
import websockets
from websockets.asyncio.client import connect

def handle_post_request(endpoint, data=None, api_key=None):
    try:
        response = requests.post(
            SERVER_URL + endpoint,
            headers={"x-api-key": api_key},
            json=data,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 502:
            raise SystemExit(f"Onyx Engine API error: {e}")
        try:
            error_data = json.loads(response.text)
            error = error_data.get('detail', response.text or str(e))
        except (json.JSONDecodeError, ValueError):
            error = response.text or str(e)
        raise SystemExit(f"Onyx Engine API error: {error}")
    except requests.exceptions.ConnectionError:
        raise SystemExit("Onyx Engine API error: Unable to connect to the server.")
    except requests.exceptions.Timeout:
        raise SystemExit("Onyx Engine API error: The request connection timed out.")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Onyx Engine API error: An unexpected error occurred: {e}")
        
    return response.json()

def upload_object(filename, object_type, object_id, api_key=None):
    # Get secure upload URL from the cloud
    response = handle_post_request("/generate_upload_url", {"object_filename": filename, "object_type": object_type, "object_id": object_id}, api_key=api_key)

    # Upload the object using the secure URL
    local_copy_path = os.path.join(ONYX_PATH, object_type + 's', filename)
    file_size = os.path.getsize(local_copy_path)
    with tqdm(total=file_size, desc=f'{filename}', unit="B", bar_format="{percentage:.1f}% |{bar}| {desc} | {rate_fmt}", unit_scale=True, unit_divisor=1024) as progress_bar:
        with open(local_copy_path, "rb") as file:
            fields = response["fields"]
            fields["file"] = (filename, file)
            e = MultipartEncoder(fields=fields)
            m = MultipartEncoderMonitor(e, lambda monitor: progress_bar.update(monitor.bytes_read - progress_bar.n))
            headers = {"Content-Type": m.content_type}
            try:
                response = requests.post(response['url'], data=m, headers=headers)
                response.raise_for_status()
                progress_bar.n = progress_bar.total
            except requests.exceptions.HTTPError as e:
                try:
                    error_data = json.loads(response.text)
                    error = error_data.get('detail', response.text or str(e))
                except (json.JSONDecodeError, ValueError):
                    error = response.text or str(e)
                raise SystemExit(f"Onyx Engine API error: {error}")
            except requests.exceptions.ConnectionError:
                raise SystemExit("Onyx Engine API error: Unable to connect to the server.")
            except requests.exceptions.Timeout:
                raise SystemExit("Onyx Engine API error: The request connection timed out.")
            except requests.exceptions.RequestException as e:
                raise SystemExit(f"Onyx Engine API error: An unexpected error occurred: {e}")
            
def upload_object_multipart(
    filename,
    object_type,
    dataset_id,
    upload_plan,
    api_key=None,
):
    local_filename = os.path.join(ONYX_PATH, object_type + 's', filename)
    file_size = os.path.getsize(local_filename)
    part_size = upload_plan["part_size"]
    parts = []
    abort_required = False
    upload_id = upload_plan["upload_id"]
    stream_chunk_size = 256 * 1024

    class StreamingBody:
        def __init__(self, file_handle, length, chunk_size, progress):
            self.file_handle = file_handle
            self.length = length
            self.chunk_size = chunk_size
            self.progress = progress
            self.remaining = length

        def __iter__(self):
            return self

        def __next__(self):
            if self.remaining <= 0:
                raise StopIteration
            data = self.file_handle.read(min(self.chunk_size, self.remaining))
            if not data:
                raise SystemExit("Onyx Engine API error: Unexpected EOF while reading upload part.")
            self.remaining -= len(data)
            self.progress.update(len(data))
            return data

        def __len__(self):
            return self.length

    with tqdm(total=file_size, desc=f'{filename}', unit="B", bar_format="{percentage:.1f}% |{bar}| {desc} | {rate_fmt}", unit_scale=True, unit_divisor=1024) as progress_bar:
        try:
            with open(local_filename, "rb") as file:
                for part in upload_plan["parts"]:
                    start = (part["part_number"] - 1) * part_size
                    end = min(start + part_size, file_size)
                    part_length = end - start
                    file.seek(start)
                    url = part["url"]
                    response = requests.put(
                        url,
                        data=StreamingBody(file, part_length, stream_chunk_size, progress_bar),
                    )
                    response.raise_for_status()
                    etag = response.headers.get("ETag") or response.headers.get("etag")
                    if etag is None:
                        raise SystemExit("Onyx Engine API error: Missing ETag for multipart upload.")

                    parts.append({"part_number": part["part_number"], "etag": etag})
        except requests.exceptions.HTTPError as e:
            abort_required = True
            try:
                error_text = e.response.text if e.response is not None else str(e)
                error_data = json.loads(error_text)
                error = error_data.get('detail', error_text or str(e))
            except (json.JSONDecodeError, ValueError):
                error = e.response.text if e.response is not None else str(e)
            raise SystemExit(f"Onyx Engine API error: {error}")
        except requests.exceptions.ConnectionError:
            abort_required = True
            raise SystemExit("Onyx Engine API error: Unable to connect to the server.")
        except requests.exceptions.Timeout:
            abort_required = True
            raise SystemExit("Onyx Engine API error: The request connection timed out.")
        except requests.exceptions.RequestException as e:
            abort_required = True
            raise SystemExit(f"Onyx Engine API error: An unexpected error occurred: {e}")
        finally:
            if abort_required:
                try:
                    handle_post_request(
                        "/abort_dataset_upload",
                        {
                            "dataset_id": dataset_id,
                            "file_path": filename,
                            "upload_id": upload_id,
                        },
                        api_key=api_key
                    )
                except Exception:
                    pass

    return parts

def download_object(filename, object_type, object_id: Optional[str] = None, api_key=None):
    # Get secure download URL from the cloud
    response = handle_post_request("/generate_download_url", {"object_filename": filename, "object_type": object_type, "object_id": object_id}, api_key=api_key)
    download_url = response["download_url"]

    # Download the object using the secure URL
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            error_data = json.loads(response.text)
            error = error_data.get('detail', response.text or str(e))
        except (json.JSONDecodeError, ValueError):
            error = response.text or str(e)
        raise SystemExit(f"Onyx Engine API error: {error}")
    except requests.exceptions.ConnectionError:
        raise SystemExit("Onyx Engine API error: Unable to connect to the server.")
    except requests.exceptions.Timeout:
        raise SystemExit("Onyx Engine API error: The request connection timed out.")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Onyx Engine API error: An unexpected error occurred: {e}")

    # Write the object to local storage
    block_size = 1024 * 64
    total_size = int(response.headers.get("content-length", 0))
    local_copy_path = os.path.join(ONYX_PATH, object_type + 's', filename)
    with tqdm(total=total_size, desc=f'{filename}', unit="B", bar_format="{percentage:.1f}% |{bar}| {desc} | {rate_fmt}", unit_scale=True) as progress_bar:
        with open(local_copy_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

class SourceObject(BaseModel):
    name: str
    id: Optional[str] = None

def set_object_metadata(object_name, object_type, object_config, object_sources: List[SourceObject]=[], api_key=None):
    # Request to set metadata for the object in onyx engine
    metadata = handle_post_request("/set_object_metadata", {
        "object_name": object_name,
        "object_type": object_type,
        "object_config": object_config,
        "object_sources": [source.model_dump() for source in object_sources],
    }, api_key=api_key)
    if metadata is None:
        return None
    if isinstance(metadata['config'], str):
        metadata['config'] = json.loads(metadata['config'])
 
    return metadata
    
async def monitor_training_job(job_id: str, training_config: TrainingConfig, api_key=None):
    headers = {
        "x-api-key": api_key,
    }
    query_params = {
        "client": "api",
        "job-id": job_id
    }
    query_string = "&".join([f"{key}={value}" for key, value in query_params.items()])
    try:
        async with connect(f"{WSS_URL}/monitor_training?{query_string}", additional_headers=headers) as websocket:
            total_iters = training_config.training_iters
            with tqdm(total=total_iters, desc="Training: ", bar_format="{desc}{percentage:.1f}%|{bar}|{n_fmt}/{total_fmt} train_iters") as pbar:
                while True:
                    train_update = json.loads(await websocket.recv())
                    if train_update.get('status') == 'training_complete':
                        pbar.n = total_iters
                        pbar.refresh()
                        break
                    
                    pbar.n = train_update['train_iter']
                    pbar.refresh()
            print("Training completed.")
            
    except websockets.exceptions.ConnectionClosedOK as e:
        print(f"Training monitor connection closed.")
        return
    except websockets.exceptions.WebSocketException as e:
        raise SystemExit(f"Onyx Engine API error: {e}")
