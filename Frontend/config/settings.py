import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL")

if not BACKEND_URL:
    raise RuntimeError("BACKEND_URL is not set in the environment")