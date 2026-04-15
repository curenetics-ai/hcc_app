import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient
from main import app 

client = TestClient(app)

def test_health_endpoint():

    response = client.get('/api/v1/health')

    assert response.status_code ==200

    json_data = response.json()

    print(json_data)

    assert 'status' in json_data
    assert 'model_loaded' in json_data 
    assert 'database_connected' in json_data