"""Config router — exposes runtime flags to the frontend."""

import os
from fastapi import APIRouter

router = APIRouter()

@router.get("")
def get_config():
    return {
        "demo_mode": os.getenv("DEMO_MODE", "false").lower() == "true",
        "version": "1.0.0",
    }
