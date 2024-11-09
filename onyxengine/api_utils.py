import os
import json
import requests
from tqdm import tqdm
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from onyxengine import SERVER_URL, ONYX_API_KEY, ONYX_PATH

def handle_post_request(endpoint, data=None):
    try:
        response = requests.post(
            SERVER_URL + endpoint,
            headers={"x-api-key": ONYX_API_KEY},
            json=data,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 502:
            raise SystemExit(f"Onyx Engine API error: {e}")
        error = json.loads(response.text)['detail']
        raise SystemExit(f"Onyx Engine API error: {error}")
    except requests.exceptions.ConnectionError:
        raise SystemExit("Onyx Engine API error: Unable to connect to the server.")
    except requests.exceptions.Timeout:
        raise SystemExit("Onyx Engine API error: The request connection timed out.")
    except requests.exceptions.RequestException:
        raise SystemExit("Onyx Engine API error: An unexpected error occurred.", e)
        
    return response.json()

def upload_object(filename, object_type):
    # Get secure upload URL from the cloud
    response = handle_post_request("/generate_upload_url", {"object_filename": filename, "object_type": object_type})

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
                error = json.loads(response.text)['detail']
                raise SystemExit(f"Onyx Engine API error: {error}")
            except requests.exceptions.ConnectionError:
                raise SystemExit("Onyx Engine API error: Unable to connect to the server.")
            except requests.exceptions.Timeout:
                raise SystemExit("Onyx Engine API error: The request connection timed out.")
            except requests.exceptions.RequestException:
                raise SystemExit("Onyx Engine API error: An unexpected error occurred.", e)

def download_object(filename, object_type):
    # Get secure download URL from the cloud
    response = handle_post_request("/generate_download_url", {"object_filename": filename, "object_type": object_type})
    download_url = response["download_url"]

    # Download the object using the secure URL
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error = json.loads(response.text)['detail']
        raise SystemExit(f"Onyx Engine API error: {error}")
    except requests.exceptions.ConnectionError:
        raise SystemExit("Onyx Engine API error: Unable to connect to the server.")
    except requests.exceptions.Timeout:
        raise SystemExit("Onyx Engine API error: The request connection timed out.")
    except requests.exceptions.RequestException:
        raise SystemExit("Onyx Engine API error: An unexpected error occurred.", e)

    # Write the object to local storage
    block_size = 1024
    total_size = int(response.headers.get("content-length", 0))
    local_copy_path = os.path.join(ONYX_PATH, object_type + 's', filename)
    with tqdm(total=total_size, desc=f'{filename}', unit="B", bar_format="{percentage:.1f}% |{bar}| {desc} | {rate_fmt}", unit_scale=True) as progress_bar:
        with open(local_copy_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

def set_object_metadata(object_name, object_type, object_config, object_source_names=[]):
    # Request to set metadata for the object in onyx engine
    response = handle_post_request("/set_object_metadata", {
        "object_name": object_name,
        "object_type": object_type,
        "object_config": object_config,
        "object_source_names": object_source_names,
    })