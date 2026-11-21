import sys
from pathlib import Path
from fastapi.responses import JSONResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app

@app.middleware("http")
async def handle_vercel_path(request, call_next):
    target_path = None
    matched_path = request.headers.get("x-matched-path")
    if matched_path:
        target_path = matched_path.split("?")[0]
    elif "__path" in request.query_params:
        target_path = request.query_params["__path"]

    if target_path and target_path != request.scope.get("path"):
        request.scope["path"] = target_path

    if "debug" in request.query_params or request.scope.get("path") in ["/api/debug", "/debug"]:
        return JSONResponse({
            "resolved_scope_path": request.scope.get("path"),
            "x_matched_path": request.headers.get("x-matched-path"),
            "param_path": request.query_params.get("__path")
        })
    return await call_next(request)
