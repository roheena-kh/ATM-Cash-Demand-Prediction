import pyodbc

import pyodbc

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
    
 

