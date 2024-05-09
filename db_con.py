import sqlite3

# Connect to the SQLite database file
conn = sqlite3.connect('db.sqlite3')

# Create a cursor object
cursor = conn.cursor()

# Execute a query (sample from me)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# Fetch results
tables = cursor.fetchall()
print(tables)

# Close the connection
conn.close()
