# celery_app.py
# This is the Celery application configuration.
# Every worker and every task imports from here.
# Think of this as the "config.py" for the async infrastructure.

from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Redis is our broker — holds tasks waiting to be processed
# Redis is also our backend — stores task results for status checking later
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "insurance_pipeline",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"]  # tells Celery where to find task definitions
)

# Configuration
app.conf.update(
    # Serialize tasks as JSON — human readable, debuggable
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # If a task fails, retry up to 3 times with 60 second delay
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Timezone
    timezone="Europe/Berlin",

    # Result expiry — keep task results for 24 hours then clean up
    result_expires=86400,
)