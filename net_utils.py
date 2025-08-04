import aiohttp
import ssl
import certifi
import json
import atexit
import asyncio

# Single TLS context with system CAs
_ssl_ctx = ssl.create_default_context(cafile=certifi.where())

# Global session - will be lazily initialized
_shared_session = None

class SharedSession:
    """Lazy session that creates the aiohttp session only when needed"""
    
    def __init__(self):
        self._session = None
    
    def _ensure_session(self):
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit_per_host=500,              # was 100
                limit=2000,                      # total connection limit
                ssl=_ssl_ctx,                    # HTTP/2 capable
                enable_cleanup_closed=True,
                keepalive_timeout=60,
                force_close=False,               # reuse connections
                ttl_dns_cache=300,              # DNS caching
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
                trust_env=True,                      # proxy-friendly
                json_serialize=lambda x: json.dumps(x, separators=(",", ":")),
            )
        return self._session
    
    def __getattr__(self, name):
        # Delegate all attributes to the actual session
        return getattr(self._ensure_session(), name)
    
    async def __aenter__(self):
        return await self._ensure_session().__aenter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            return await self._session.__aexit__(exc_type, exc_val, exc_tb)
    
    async def close(self):
        """Properly close the session and connector"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self._session and not self._session.closed:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except:
                pass  # Ignore errors during cleanup

# Global shared session instance
shared_session = SharedSession()

# Cleanup function for atexit
def _cleanup_shared_session():
    """Cleanup function called at program exit"""
    if shared_session._session and not shared_session._session.closed:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(shared_session.close())
            loop.close()
        except:
            pass  # Ignore errors during cleanup

# Register cleanup function
atexit.register(_cleanup_shared_session)
