
import sys
import os
from flask import Flask

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from app import app, dashboard_data
    
    print("Successfully imported app")
    
    with app.app_context():
        try:
            print("Attempting to call dashboard_data()...")
            # We can call the view function directly if we mock the request context or if it doesn't depend on request
            # dashboard_data uses _get_conn_cursor which uses db_connect
            
            # Mocking request context just in case, though dashboard_data doesn't seem to use 'request' object directly except maybe for session?
            # It uses user_id=1 hardcoded in queries.
            
            response = dashboard_data()
            print("Response:", response)
            if hasattr(response, 'get_json'):
                print("JSON:", response.get_json())
            elif isinstance(response, tuple):
                 print("Response Tuple:", response)
            
        except Exception as e:
            print("Error calling dashboard_data:")
            import traceback
            traceback.print_exc()

except ImportError as e:
    print(f"Failed to import app: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
