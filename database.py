import sqlite3

# ---------------------------
# CONNECT
# ---------------------------
def create_connection():
    return sqlite3.connect("pricing.db")

# ---------------------------
# CREATE TABLES
# ---------------------------
def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pricing_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        original_price REAL,
        recommended_price REAL,
        demand REAL,
        reason TEXT
    )
    """)

    conn.commit()
    conn.close()

# ---------------------------
# USER FUNCTIONS
# ---------------------------
def add_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users VALUES (?, ?)", (username, password))
        conn.commit()
    except:
        pass
    conn.close()

def validate_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

# ---------------------------
# HISTORY FUNCTIONS
# ---------------------------
def save_history(username, original_price, recommended_price, demand, reason):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO pricing_history (username, original_price, recommended_price, demand, reason)
    VALUES (?, ?, ?, ?, ?)
    """, (username, original_price, recommended_price, demand, reason))

    conn.commit()
    conn.close()

def get_history(username):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM pricing_history WHERE username=?", (username,))
    rows = cursor.fetchall()

    conn.close()
    return rows