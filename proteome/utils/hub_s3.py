import os
from typing import Any, Dict, Optional

import boto3
import torch.hub
from botocore import UNSIGNED
from botocore.client import Config
from torch.serialization import MAP_LOCATION

__all__ = [
    "download_s3_url_to_file",
    "load_state_dict_from_s3_url",
]


def download_s3_url_to_file(
    s3_url: str,
    dst: str,
    boto_client_config: Config = Config(signature_version=UNSIGNED),
) -> None:
    r"""Download object at the given S3 URL to a local path.

    Args:
        s3_url (str): S3 URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        boto_client_config (botocore.client.Config):
            boto3 client configuration. By default uses --no-sign-request config.

    """
    if not s3_url.startswith("s3://"):
        raise ValueError("S3 URL must start with 's3://'.")

    bucket, key = s3_url[5:].split("/", 1)
    client = boto3.client("s3", config=boto_client_config)

    with open(dst, mode="wb") as f:
        client.download_fileobj(Bucket=bucket, Key=key, Fileobj=f)


def load_state_dict_from_s3_url(
    s3_url: str,
    model_dir: Optional[str] = None,
    map_location: MAP_LOCATION = None,
    boto_client_config: Config = Config(signature_version=UNSIGNED),
) -> Dict[str, Any]:
    r"""Loads the Torch serialized object at the given S3 URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        s3_url (str): S3 URL of the object to download
        model_dir (str, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        boto_client_config (botocore.client.Config):
            boto3 client configuration. By default uses --no-sign-request config.

    Returns:
        state_dict (dict): a dict containing the state of the pytorch model

    """
    if not s3_url.startswith("s3://"):
        raise ValueError("S3 URL must start with 's3://'.")

    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, os.path.basename(s3_url))
    if not os.path.exists(cached_file):
        print(f"Downloading model from {s3_url}...")
        download_s3_url_to_file(
            s3_url, cached_file, boto_client_config=boto_client_config
        )
        print(f"Model downloaded and cached in {cached_file}.")

    return torch.load(cached_file, map_location=map_location)
