# FASTA > ALPHAFOLD DB > DRIVE
from Bio import SeqIO
import os
import requests
import time
import json

input_fasta = TRAIN_SEQUENCES
base_directory = INPUT_DATA

def retrieve_files(input_fasta, base_directory, last_downloaded_file, delay=0.1, max_retries=5):
    found_last_downloaded = False
    for record in SeqIO.parse(input_fasta, 'fasta'):
        uniprot_id = record.id

        # If this is where we left off last time, start processing
        if last_downloaded_file is not None and uniprot_id == last_downloaded_file:
            found_last_downloaded = True
        elif last_downloaded_file is not None and not found_last_downloaded:
            continue

        pdb_file_name = f'{uniprot_id}.pdb'
        pae_file_name = f'{uniprot_id}.json'

        uniprot_directory = os.path.join(base_directory, uniprot_id)
        pdb_file_path = os.path.join(uniprot_directory, pdb_file_name)
        pae_file_path = os.path.join(uniprot_directory, pae_file_name)

        # Retrieve PDB, JSON file
        pdb_url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
        pae_url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v4.json'

        # Check if files exist in Alphafold database
        pdb_exists = requests.head(pdb_url).status_code == 200
        pae_exists = requests.head(pae_url).status_code == 200

        # Only create the directory if one of the files exists
        if pdb_exists or pae_exists:
            if not os.path.exists(uniprot_directory):
                os.makedirs(uniprot_directory)

            # Only retrieve the file if it exists
            if pdb_exists:
                retrieve_file(pdb_url, pdb_file_path, 'PDB', delay, max_retries, uniprot_id, base_directory)
            if pae_exists:
                retrieve_file(pae_url, pae_file_path, 'JSON', delay, max_retries, uniprot_id, base_directory)

def retrieve_file(url, file_path, file_type, delay, max_retries, uniprot_id, base_directory):
    if os.path.exists(file_path):
        print(f'{file_type} file {file_path} already exists. Skipping this entry.')
        return

    retries = 0
    while retries <= max_retries:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'w') as file:
                if file_type == 'PDB':
                    file.write(response.text)
                elif file_type == 'JSON':
                    json.dump(response.json(), file)
            # Record the successfully downloaded file
            with open(os.path.join(base_directory, 'last_downloaded.txt'), 'w') as f:
                f.write(uniprot_id)
            break
        elif response.status_code == 429:
            wait_time = (2 ** retries) * delay
            print(f'Received 429 status code from server. Waiting for {wait_time} seconds and retrying...')
            time.sleep(wait_time)
            retries += 1
        else:
            print(f'{file_type} file for {url} not found in Alphafold database. Skipping this entry.')
            break
    time.sleep(delay)

last_downloaded_file = None
if os.path.exists(os.path.join(base_directory, 'last_downloaded.txt')):
    with open(os.path.join(base_directory, 'last_downloaded.txt'), 'r') as f:
        last_downloaded_file = f.read().strip()

max_retries = 10
retries = 0
success = False

while not success and retries < max_retries:
    try:
        retrieve_files(input_fasta, base_directory, last_downloaded_file)
        success = True
    except Exception as e:
        retries += 1
        print(f'Attempt {retries} failed with error {str(e)}. Retrying in {2} seconds...')
        time.sleep(2)

if not success:
    print("Operation failed after maximum retries. Please check the error.")
else:
    print("Operation succeeded.")
