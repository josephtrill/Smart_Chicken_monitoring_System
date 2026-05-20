
import sys
import os
from flask import session

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from app import app
    
    print("Successfully imported app")
    
    app.config['TESTING'] = True
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'test_user'
            sess['_last_active'] = '2025-01-01T00:00:00' # Future date or recent

        print("Attempting to access dashboard route / ...")
        response = client.get('/')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Dashboard loaded successfully.")
        elif response.status_code == 302:
            print(f"Redirected to: {response.headers['Location']}")
        else:
            print("Failed to load dashboard.")
            print(response.data.decode('utf-8'))

except ImportError as e:
    print(f"Failed to import app: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
