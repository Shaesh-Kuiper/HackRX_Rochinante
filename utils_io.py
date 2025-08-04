# utils_io.py (new)
import requests
import tempfile
import pathlib
import shutil
import aiohttp
import asyncio
import aiofiles
import os

async def async_download_blob(url: str, session: aiohttp.ClientSession) -> pathlib.Path:
    """Async stream‑download the blob URL to a temp file and return its Path."""
    # Handle local file:// URLs for testing
    if url.startswith("file://"):
        local_path = pathlib.Path(url[7:])  # Remove "file://" prefix
        if local_path.exists():
            # Copy to temp file to maintain consistent interface
            suffix = local_path.suffix
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            shutil.copy2(local_path, tmp_path)
            return pathlib.Path(tmp_path)
        else:
            raise FileNotFoundError(f"Local file not found: {local_path}")
    
    # Handle remote URLs
    suffix = pathlib.Path(url.split("?")[0]).suffix  # keep extension without query params
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        async with session.get(url, timeout=30) as r:
            r.raise_for_status()
            async with aiofiles.open(tmp_path, "wb") as f:
                async for chunk in r.content.iter_chunked(8192):
                    await f.write(chunk)
        return pathlib.Path(tmp_path)
    except Exception:
        os.unlink(tmp_path)
        raise

def download_blob(url: str) -> pathlib.Path:
    """Stream‑download the blob URL to a temp file and return its Path."""
    # Handle local file:// URLs for testing
    if url.startswith("file://"):
        local_path = pathlib.Path(url[7:])  # Remove "file://" prefix
        if local_path.exists():
            # Copy to temp file to maintain consistent interface
            suffix = local_path.suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.close()
            shutil.copy2(local_path, tmp.name)
            return pathlib.Path(tmp.name)
        else:
            raise FileNotFoundError(f"Local file not found: {local_path}")
    
    # Handle remote URLs
    suffix = pathlib.Path(url.split("?")[0]).suffix  # keep extension without query params
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            tmp.write(chunk)
    tmp.close()
    return pathlib.Path(tmp.name)