# net_utils.py
import aiohttp
import ssl
import certifi
import json
import atexit
import asyncio
import os

# Tunables (envs give you easy runtime knobs)
DOWNLOAD_CONNECT_TIMEOUT = float(os.getenv("DL_CONNECT_TIMEOUT", "10"))
DOWNLOAD_SOCK_CONNECT_TIMEOUT = float(os.getenv("DL_SOCK_CONNECT_TIMEOUT", "10"))
DOWNLOAD_SOCK_READ_TIMEOUT = float(os.getenv("DL_SOCK_READ_TIMEOUT", "120"))
CONNECTIONS_PER_HOST = int(os.getenv("DL_CONNECTIONS_PER_HOST", "64"))
CONNECTIONS_TOTAL = int(os.getenv("DL_CONNECTIONS_TOTAL", "1024"))

_ssl_ctx = ssl.create_default_context(cafile=certifi.where())
_shared_session = None

class SharedSession:
    """Lazy session that creates the aiohttp session only when needed"""

    def __init__(self):
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit_per_host=CONNECTIONS_PER_HOST,
                limit=CONNECTIONS_TOTAL,
                ssl=_ssl_ctx,
                enable_cleanup_closed=True,
                keepalive_timeout=60,
                force_close=False,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            # IMPORTANT: total=None removes the global 30s kill-switch
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=DOWNLOAD_CONNECT_TIMEOUT,
                sock_connect=DOWNLOAD_SOCK_CONNECT_TIMEOUT,
                sock_read=DOWNLOAD_SOCK_READ_TIMEOUT,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                trust_env=True,
                json_serialize=lambda x: json.dumps(x, separators=(",", ":")),
                auto_decompress=True,  # fine for PDFs; Azure usually sends identity anyway
            )
        return self._session

    def __getattr__(self, name):
        return getattr(self._ensure_session(), name)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

# Global shared session instance
shared_session = SharedSession()

def _cleanup_shared_session():
    if shared_session._session and not shared_session._session.closed:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(shared_session.close())
            loop.close()
        except:
            pass

atexit.register(_cleanup_shared_session)
