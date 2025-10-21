# init_db.py
import sqlite3

# Connect to the database file (it will be created if it doesn't exist)
connection = sqlite3.connect('conversation_history.db')

# Create a cursor object to execute SQL commands
cursor = connection.cursor()

# Create the 'history' table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    )
''')

# Commit the changes and close the connection
connection.commit()
connection.close()

print("Database 'conversation_history.db' and 'history' table created successfully.")