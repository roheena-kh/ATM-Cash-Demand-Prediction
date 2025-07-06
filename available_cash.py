import pyodbc
from pprint import pprint

def get_available_cash():
    server = ''
    database = ''
    username = ''
    password = ''
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={server};DATABASE={database};UID={username};PWD={password}'
    )
    query = """
       SELECT [deviceid],[starttime] AS "Last_Replinishement_time",
        [total_cash] as "TotalAndReplinshedCash"
        ,[Total_cash_remaining] "AFN_Available",[Total_cash_remaining_usd] as "USD_Available" FROM [Reporting].[dbo].[Cassete_counter]
        where 
            CAST([postingdatetime] AS DATE) = CAST(CURRENT_TIMESTAMP AS DATE)
    """

    result = {}
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        for row in cursor.fetchall():
            term_id = row[0]
            afn_cash = row[3]
            usd_cash = row[4]
            result[term_id] = {
                "AFN_Available": afn_cash,
                "USD_Available": usd_cash
            }
    return result

if __name__ == "__main__":
    pprint(get_available_cash())

    
 

