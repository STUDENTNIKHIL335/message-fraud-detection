import sys
from pathlib import Path
from fastapi.responses import JSONResponse

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app

@app.middleware("http")
async def debug_middleware(request, call_next):
    if request.url.path in ["/api/debug", "/debug"]:
        return JSONResponse({
            "url_path": request.url.path,
            "scope_path": request.scope.get("path", ""),
            "scope_root_path": request.scope.get("root_path", ""),
            "routes": [getattr(r, "path", None) for r in app.routes if hasattr(r, "path")]
        })
    return await call_next(request)
