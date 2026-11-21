import sys
from pathlib import Path
from fastapi.responses import JSONResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app

@app.middleware("http")
async def inspect_every_request(request, call_next):
    if "debug" in request.url.path or request.query_params.get("debug") == "1":
        return JSONResponse({
            "url": str(request.url),
            "url_path": request.url.path,
            "scope_path": request.scope.get("path"),
            "scope_root_path": request.scope.get("root_path"),
            "scope_raw_path": request.scope.get("raw_path", b"").decode(errors="ignore"),
            "registered_routes": [getattr(r, "path", None) for r in app.routes if hasattr(r, "path")]
        })
    return await call_next(request)
