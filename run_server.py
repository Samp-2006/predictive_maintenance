# ============================================================
# run_server.py
# Convenience script to start the FastAPI server.
# Usage:  python run_server.py
# ============================================================

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import api_cfg
from utils.logger import logger

if __name__ == "__main__":
    logger.info(f"Starting API on http://{api_cfg.host}:{api_cfg.port}")
    uvicorn.run(
        "api.main:app",
        host=api_cfg.host,
        port=api_cfg.port,
        reload=api_cfg.reload,
        log_level=api_cfg.log_level,
    )
