
import sys
import os
import json
from app import app

app.config['TESTING'] = True
app.secret_key = 'test'

with app.test_client() as client:
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    payload = {
        "user_id": 1,
        "device_id": "debug_device",
        "temperature": 50.0,
        "humidity": 60.0,
        "gas": 200.0,
        "mic_rms": 0.1,
        "mic_peak": 0.2,
        "audio_level_db": 50.0
    }
    print("Sending POST to /api/sensor...")
    res = client.post('/api/sensor', data=json.dumps(payload))
    print(f"Status: {res.status_code}")
    print(f"Response: {res.data.decode('utf-8')}")
