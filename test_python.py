import csv
import json
import os

print("Python is working!")
print(f"Current directory: {os.getcwd()}")

# Test if we can read the WHO ISH chart
try:
    with open('WHO ish chart.csv', 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        print(f"Successfully read WHO ISH chart with {len(rows)} rows")
        if rows:
            print(f"Sample row: {rows[0]}")
except Exception as e:
    print(f"Error reading WHO ISH chart: {e}")

# Test if we can read the template
try:
    with open('templates/index.html', 'r') as file:
        content = file.read()
        print(f"Successfully read HTML template ({len(content)} characters)")
except Exception as e:
    print(f"Error reading HTML template: {e}")

print("All tests completed!") 