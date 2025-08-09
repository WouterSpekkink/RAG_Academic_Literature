#!/bin/bash

# One file to keep the papers that I have already ingested
# One dir to store newly added papers
# A temporary dir for image-based pdfs.
existing_file="./data/ingested.txt"
output_dir="./data/new"
temp_dir="./data/temp"

counter=0

total=$(find /home/wouter/Tools/Zotero/storage/ -type f -name "*.pdf" | wc -l)

find /home/wouter/Tools/Zotero/storage -type f -name "*.pdf" | while read -r file
do
    base_name=$(basename "$file" .pdf)

    if grep -Fxq "$base_name.txt" "$existing_file"; then
	echo -ne "Text file for $file already exists, skipping.\n"
    else 
	pdftotext -enc UTF-8 "$file" "$output_dir/$base_name.txt"
	
    fi
    counter=$((counter + 1))
    echo -ne "Processed $counter out of $total PDFs.\r"
    
done
