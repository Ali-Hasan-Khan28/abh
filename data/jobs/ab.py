import sqlite3

def view_sqlite_data(database_file):
    """
    Displays the data within a SQLite database file.

    Args:
        database_file (str): The path to the SQLite database file.
    """

    try:
        # Establish the connection
        conn = sqlite3.connect(database_file)

        # Create a cursor for database interactions 
        cursor = conn.cursor()

        # Get a list of table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(tables)
        # Iterate over each table
        for table_name in tables:
            # print(f"\n*Table: {table_name[0]}*")

            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name[0]})")
            columns = [col[1] for col in cursor.fetchall()]
            print(columns)
            break
            # Fetch and display the data
            cursor.execute(f"SELECT * FROM {table_name[0]}")
            rows = cursor.fetchall()
            for row in rows:
                for col_name, value in zip(columns, row):
                    pass
                    # print(f"  {col_name}: {value}")

        # Close the connection
        conn.close()

    except sqlite3.Error as e:
        print(f"Error occurred while working with '{database_file}': {e}")


# View data from both files:
view_sqlite_data("job-suche_en.sqlite")
# view_sqlite_data("job-suche-en.sqlite")