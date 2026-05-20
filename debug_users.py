
import sys
import os
from app import _get_conn_cursor

sys.path.append(os.getcwd())

conn, cur = _get_conn_cursor(dictionary=True)
cur.execute("SELECT * FROM users")
rows = cur.fetchall()
print(f"Found {len(rows)} users:")
for r in rows:
    print(f"ID: {r['id']}, Username: {r['username']}, Email: {r['email']}")
cur.close()
conn.close()
