
import sys
import os
from flask import session
from datetime import datetime

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from app import app
    
    print("Successfully imported app")
    
    app.config['TESTING'] = True
    app.secret_key = 'fixed_key_for_testing' # Ensure consistent key
    
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'test_user'
            sess['_last_active'] = datetime.utcnow().isoformat()

        print("Attempting to access dashboard route / ...")
        response = client.get('/')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Dashboard loaded successfully.")
        elif response.status_code == 302:
            print(f"Redirected to: {response.headers['Location']}")
            # Follow redirect
            response = client.get(response.headers['Location'])
            print(f"Followed redirect status: {response.status_code}")
        elif response.status_code == 500:
            print("Server Error (500)!")
            # In testing mode, we might see the error
            print(response.data.decode('utf-8'))
        else:
            print(f"Failed to load dashboard. Status: {response.status_code}")

except ImportError as e:
    print(f"Failed to import app: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
