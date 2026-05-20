
import requests
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_flow():
    s = requests.Session()
    
    # 1. Login
    print("Attempting login...")
    try:
        # Assuming default credentials or I can create a user if needed.
        # But let's try to just hit the dashboard and see if it redirects (302) or errors (500)
        r = s.get(f"{BASE_URL}/")
        print(f"GET / status: {r.status_code}")
        if r.status_code == 500:
            print("Server Error on root /")
            print(r.text)
            return

        if r.url.endswith("/login"):
            print("Redirected to login page. Logging in...")
            # Try to login with admin/admin or similar if I knew credentials.
            # But I can check if the login page itself loads.
            r = s.get(f"{BASE_URL}/login")
            print(f"GET /login status: {r.status_code}")
            if r.status_code == 500:
                print("Server Error on /login")
                print(r.text)
                return
            
            # Try to login
            # I need a valid user. I'll assume 'admin' exists or create one via app function if I could.
            # For now, let's just see if the pages load without crashing.
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Is it running?")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_flow()
