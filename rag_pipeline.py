# rag_pipeline.py
import os
import asyncio
from typing import List, Tuple
from urllib.parse import urlparse
import hashlib
import pathlib
import math
import shutil

import net_utils
from V8_api import run_rag_pipeline_v8

CACHE_DIR = os.path.join(os.getcwd(), "cache_pdfs")
os.makedirs(CACHE_DIR, exist_ok=True)

# Tunables
MAX_FILE_CONCURRENCY = int(os.getenv("DL_MAX_FILE_CONCURRENCY", "24"))   # how many files at once
MAX_SEGMENTS_PER_FILE = int(os.getenv("DL_MAX_SEGMENTS_PER_FILE", "8"))  # parallel ranges per file
SEGMENT_SIZE_MB = int(os.getenv("DL_SEGMENT_SIZE_MB", "8"))              # each range size
SEGMENT_SIZE = SEGMENT_SIZE_MB * 1024 * 1024
STREAM_CHUNK_SIZE = int(os.getenv("DL_STREAM_CHUNK", str(1 * 1024 * 1024)))  # 1MB

def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def _is_azure_blob(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.netloc.endswith(".blob.core.windows.net")
    except Exception:
        return False

def _safe_pdf_filename(src: str) -> str:
    base = pathlib.Path(urlparse(src).path).name or "document.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    h = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
    return f"{h}_{base}"

async def _download_stream(url: str, dest_path: str) -> str:
    tmp = dest_path + ".downloading"
    # stream to disk
    async with net_utils.shared_session.get(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to download {url}: HTTP {resp.status}")
        with open(tmp, "wb") as f:
            async for chunk in resp.content.iter_chunked(STREAM_CHUNK_SIZE):
                if not chunk:
                    break
                f.write(chunk)
    os.replace(tmp, dest_path)
    return dest_path

async def _head_content_length(url: str) -> tuple[int|None, bool]:
    try:
        async with net_utils.shared_session.head(url, allow_redirects=True) as h:
            cl = h.headers.get("Content-Length")
            size = int(cl) if cl is not None else None
            accept_ranges = (h.headers.get("Accept-Ranges", "") or "").lower() == "bytes"
            # Azure sometimes omits Accept-Ranges, but still supports Range; treat unknown as True
            if _is_azure_blob(url):
                accept_ranges = True if size else accept_ranges
            return size, accept_ranges
    except Exception:
        return None, False

async def _fetch_range(url: str, start: int, end: int, part_path: str):
    headers = {"Range": f"bytes={start}-{end}"}
    async with net_utils.shared_session.get(url, headers=headers) as resp:
        if resp.status not in (200, 206):
            raise RuntimeError(f"Range {start}-{end} failed for {url}: HTTP {resp.status}")
        with open(part_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(STREAM_CHUNK_SIZE):
                if not chunk:
                    break
                f.write(chunk)

async def _download_segmented(url: str, dest_path: str, total_size: int):
    # Decide number of segments
    seg_size = max(SEGMENT_SIZE, 1 * 1024 * 1024)
    n_segs = math.ceil(total_size / seg_size)
    n_segs = min(n_segs, MAX_SEGMENTS_PER_FILE)

    part_paths = [f"{dest_path}.part{i}" for i in range(n_segs)]
    tasks = []
    start = 0
    for i in range(n_segs):
        end = min(start + seg_size - 1, total_size - 1)
        tasks.append(asyncio.create_task(_fetch_range(url, start, end, part_paths[i])))
        start = end + 1

    # Run with a per-file concurrency gate to avoid stampeding
    # (we keep it simple: tasks already created, but you can wrap with a Semaphore if you want stricter caps)
    await asyncio.gather(*tasks)

    # Merge parts atomically
    tmp = dest_path + ".downloading"
    with open(tmp, "wb") as out:
        for p in part_paths:
            with open(p, "rb") as partf:
                shutil.copyfileobj(partf, out, STREAM_CHUNK_SIZE)
            os.remove(p)
    os.replace(tmp, dest_path)
    return dest_path

async def _download_one_http(url: str, dest_path: str) -> str:
    # If already downloaded (nonzero), skip
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    size, ranges_ok = await _head_content_length(url)
    # If we know it's large and ranges are OK → parallel ranges
    if size and size >= 2 * SEGMENT_SIZE and ranges_ok:
        return await _download_segmented(url, dest_path, size)
    # Fallback: single stream
    return await _download_stream(url, dest_path)

async def _download_one_azure(url: str, dest_path: str) -> str:
    from azure.storage.blob.aio import BlobClient  # requires azure-storage-blob

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    tmp = dest_path + ".downloading"
    max_concurrency = int(os.getenv("AZ_BLOB_MAX_CONCURRENCY", str(MAX_SEGMENTS_PER_FILE)))

    async with BlobClient.from_blob_url(url) as bc:
        downloader = await bc.download_blob(max_concurrency=max_concurrency)  # ← no chunk_size
        with open(tmp, "wb") as f:
            async for chunk in downloader.chunks():  # ← yields appropriately sized chunks
                if chunk:
                    f.write(chunk)

    os.replace(tmp, dest_path)
    return dest_path

async def _download_one(url: str, dest_path: str) -> str:
    if _is_azure_blob(url):
        try:
            return await _download_one_azure(url, dest_path)
        except ModuleNotFoundError:
            pass  # fall back to HTTP
        except Exception as e:
            print(f"[azure-sdk-fallback] {e!r}")  # keep logging, then fall back
    return await _download_one_http(url, dest_path)

async def _materialize_docs(document_urls: List[str]) -> List[str]:
    local_paths: List[str] = []
    downloads: List[Tuple[str, str]] = []

    for src in document_urls:
        src = src.strip()
        if _is_url(src):
            fname = _safe_pdf_filename(src)
            dest = os.path.join(CACHE_DIR, fname)
            downloads.append((src, dest))
            local_paths.append(dest)
        else:
            if not os.path.exists(src):
                raise FileNotFoundError(f"Document not found: {src}")
            local_paths.append(src)

    if downloads:
        # Wide but controlled fanout; tune with DL_MAX_FILE_CONCURRENCY
        sem = asyncio.Semaphore(MAX_FILE_CONCURRENCY)

        async def _guarded(u, d):
            async with sem:
                return await _download_one(u, d)

        await asyncio.gather(*[_guarded(u, d) for u, d in downloads])

    # After downloads finish, verify:
    for p in local_paths:
        if _is_url(p):
            # URLs aren't local paths; we only appended locals earlier
            continue
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            raise RuntimeError(f"Downloaded file missing or empty: {p}")

    return local_paths

async def answer_questions(document_urls: List[str], questions: List[str]) -> List[str]:
    local_pdf_paths = await _materialize_docs(document_urls)
    api_key = os.getenv("OPENAI_API_KEY")
    results = await run_rag_pipeline_v8(local_pdf_paths, questions, api_key)
    answers = [r.get("answer", "").strip() for r in results]
    return answers
