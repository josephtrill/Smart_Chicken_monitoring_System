
import sys
import os
import time
import json
import requests
from datetime import datetime

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from app import app, _get_conn_cursor, create_alert
    
    print("Successfully imported app")
    
    app.config['TESTING'] = True
    app.secret_key = 'fixed_key_for_testing'
    
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'test_user'
            sess['_last_active'] = datetime.utcnow().isoformat()

        # 1. Check existing alerts
        print("\n--- 1. Checking Existing Alerts in DB ---")
        conn, cur = _get_conn_cursor(dictionary=True)
        cur.execute("SELECT * FROM alerts WHERE user_id=1 ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
        print(f"Found {len(rows)} alerts.")
        for r in rows:
            print(f" - [{r['id']}] {r['type']}: {r['message']} (resolved={r['resolved']})")
        cur.close(); conn.close()

        # 2. Manually insert a test alert
        print("\n--- 2. Manually Inserting Test Alert ---")
        try:
            aid = create_alert(1, "Test", "This is a manual test alert", "warning")
            print(f"Created manual alert ID: {aid}")
        except Exception as e:
            print(f"Failed to create manual alert: {e}")

        # 3. Check dashboard_data
        print("\n--- 3. Checking /api/dashboard_data ---")
        res = client.get('/api/dashboard_data')
        if res.status_code == 200:
            data = res.json
            alerts = data.get('alerts', [])
            print(f"API returned {len(alerts)} alerts.")
            found_manual = False
            for a in alerts:
                print(f" - API Alert: {a.get('type')} - {a.get('message')}")
                if a.get('message') == "This is a manual test alert":
                    found_manual = True
            
            if found_manual:
                print("SUCCESS: Manual alert found in API response.")
            else:
                print("FAILURE: Manual alert NOT found in API response.")
        else:
            print(f"API call failed: {res.status_code}")
            print(res.data)

        # 4. Simulate Sensor Anomaly
        print("\n--- 4. Simulating Sensor Anomaly (Temp=50) ---")
        sensor_payload = {
            "user_id": 1,
            "device_id": "debug_device",
            "temperature": 50.0, # High temp
            "humidity": 60.0,
            "gas": 200.0,
            "mic_rms": 0.1,
            "mic_peak": 0.2,
            "audio_level_db": 50.0
        }
        res = client.post('/api/sensor', data=json.dumps(sensor_payload))
        print(f"Sensor POST status: {res.status_code}")
        print(f"Sensor POST response: {res.data.decode('utf-8')}")

        # Wait a bit for fusion loop (it runs every FUSION_INTERVAL, default 10s)
        print("Waiting 12 seconds for fusion loop...")
        time.sleep(12)

        # 5. Check dashboard_data again
        print("\n--- 5. Checking /api/dashboard_data after anomaly ---")
        res = client.get('/api/dashboard_data')
        alert_id_to_resolve = None
        if res.status_code == 200:
            data = res.json
            alerts = data.get('alerts', [])
            print(f"API returned {len(alerts)} alerts.")
            found_fusion = False
            for a in alerts:
                print(f" - API Alert: [{a.get('id')}] {a.get('type')} - {a.get('message')}")
                if "Stress" in a.get('message') or "Alert" in a.get('message'):
                    found_fusion = True
                # Pick the first unresolved alert to test resolution
                if not a.get('resolved') and alert_id_to_resolve is None:
                    alert_id_to_resolve = a.get('id')
            
            if found_fusion:
                print("SUCCESS: Fusion/Sensor alert found.")
            else:
                print("FAILURE: Fusion/Sensor alert NOT found.")
        else:
            print(f"API call failed: {res.status_code}")

        # 6. Test Alert Resolution
        if alert_id_to_resolve:
            print(f"\n--- 6. Testing Alert Resolution for ID {alert_id_to_resolve} ---")
            res = client.post(f'/api/alerts/{alert_id_to_resolve}/resolve')
            print(f"Resolve POST status: {res.status_code}")
            print(f"Resolve POST response: {res.data.decode('utf-8')}")
            
            if res.status_code == 200 and res.json.get('status') == 'ok':
                print("SUCCESS: Resolve endpoint returned 'ok'.")
                
                # Verify in DB
                conn, cur = _get_conn_cursor(dictionary=True)
                cur.execute("SELECT resolved FROM alerts WHERE id=%s", (alert_id_to_resolve,))
                row = cur.fetchone()
                cur.close(); conn.close()
                
                if row and row['resolved']:
                    print(f"SUCCESS: Alert {alert_id_to_resolve} is marked as resolved in DB.")
                else:
                    print(f"FAILURE: Alert {alert_id_to_resolve} is NOT marked as resolved in DB.")
            else:
                print("FAILURE: Resolve endpoint failed.")
        else:
            print("\n--- 6. Skipping Resolution Test (No alert found to resolve) ---")

except ImportError as e:
    print(f"Failed to import app: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
