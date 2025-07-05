import pyodbc

def get_available_cash():
    server = 'YOUR_SERVER'
    database = 'YOUR_DATABASE'
    username = 'YOUR_USERNAME'
    password = 'YOUR_PASSWORD'
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={server};DATABASE={database};UID={username};PWD={password}'
    )
    query = """
        SELECT TERM_ID, AVAILABLE_CASH
        FROM ATM_BALANCE_TABLE
    """
    result = {}
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        for term_id, available_cash in cursor.fetchall():
            result[term_id] = available_cash
    return result

