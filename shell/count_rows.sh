#!/bin/bash
# Usage: shell/count_rows.sh <directory>

# Directory to search for CSV files
DIR="$1"

# Initialize a counter for the total number of rows
total_rows=0

# Find all CSV files and count the number of rows in each
while IFS= read -r -d '' file; do
    # Count the number of rows in the current CSV file
    rows=$(wc -l < "$file")
    # Add the number of rows to the total count
    total_rows=$((total_rows + rows))
done < <(find "$DIR" -type f -name "*.csv" -print0)

# Print the total number of rows
echo "Total number of rows: $total_rows"
