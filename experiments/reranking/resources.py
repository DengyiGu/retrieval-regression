"""Pinned, verified downloads; all model loading uses local safetensors afterwards."""

from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path

from .data import sha256


def _download(url, target, *, limit):
    import truststore

    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    request = urllib.request.Request(url, headers={"User-Agent": "retrieval-regression-experiment"})
    temporary = target.with_suffix(target.suffix + ".part")
    size = 0
    with urllib.request.urlopen(request, context=context, timeout=60) as response:
        with temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > limit:
                    raise ValueError("download exceeds declared size limit")
                output.write(chunk)
    return temporary


def prepare_resources(config, root, *, download=False):
    for name, info in config["datasets"].items():
        path = root / "data" / f"{name}.zip"
        if not path.exists():
            if not download:
                raise ValueError(f"{name} is not cached; rerun with --download")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = _download(
                f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
                path,
                limit=50 * 1024 * 1024,
            )
            if (
                hashlib.md5(temporary.read_bytes(), usedforsecurity=False).hexdigest()
                != info["md5"]
            ):
                raise ValueError("dataset checksum mismatch")
            temporary.replace(path)
        if hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest() != info["md5"]:
            raise ValueError(f"cached {name} checksum mismatch")
    model = root / "data/models" / config["revision"]
    model.mkdir(parents=True, exist_ok=True)
    for entry in config["model_files"]:
        name = entry["rfilename"]
        if Path(name).name != name or Path(name).is_absolute():
            raise ValueError("model manifest permits only flat filenames")
        path = model / name
        if not path.exists():
            if not download:
                raise ValueError(f"model file {name} is not cached; rerun with --download")
            temporary = _download(
                f"https://huggingface.co/{config['model_id']}/resolve/{config['revision']}/{name}?download=true",
                path,
                limit=entry["size"],
            )
            _verify(temporary, entry)
            temporary.replace(path)
        _verify(path, entry)
    return model


def _verify(path, entry):
    if path.stat().st_size != entry["size"]:
        raise ValueError("model file size mismatch")
    if "lfs" in entry:
        if sha256(path) != entry["lfs"]["sha256"]:
            raise ValueError("model SHA-256 mismatch")
    else:
        data = path.read_bytes()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if digest != entry["blobId"]:
            raise ValueError("model Git blob hash mismatch")
