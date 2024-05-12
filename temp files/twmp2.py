import sqlite3
import os

# Open a connection to the SQLite database
conn = sqlite3.connect('data\jobs\job-suche_en.sqlite')
cursor = conn.cursor()

# Execute a query to select all rows from the table
cursor.execute('SELECT TITLE, * FROM JOBS_CRAWL_FILTER_JOBSUCHE_EN')  # Assuming TITLE is the first column

# Fetch all rows from the result set
rows = cursor.fetchall()

# Define the delimiter to separate values in the document file
delimiter = '\t'  # Change this to ',' if you want to use comma as the delimiter

# Specify the directory path for the document files
directory_path = 'temp files/jobs_txt/'
os.makedirs(directory_path, exist_ok=True)  # Create directory if it doesn't exist

# Iterate over the rows
for row in rows:
    # Extract the TITLE value from the row
    title = row[0]
    
    # Remove characters not allowed in file names
    title = ''.join(c for c in title if c.isalnum() or c in [' ', '-', '_'])
    
    # Specify the path to the document file using the TITLE value
    doc_file_path = os.path.join(directory_path, title + '.txt')

    # Open the document file in write mode
    with open(doc_file_path, 'w', encoding='utf-8') as doc_file:
        # Write the row to the document file
        row_str = delimiter.join(map(str, row[1:]))  # Exclude TITLE from the row string
        doc_file.write(row_str + '\n')

# Close the connection to the database
conn.close()
