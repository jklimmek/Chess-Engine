import argparse
import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from py7zr import SevenZipFile
import tqdm
import re

from .utils import *


def download_7z_file(url, save_path):
    """
    Downloads a 7z compressed file from a given URL and extracts its contents.

    Args:
        url (str): The URL of the 7z compressed file.
        save_path (str): The local path to save the downloaded and extracted files.
    """

    # Send an HTTP GET request to the specified URL with streaming content.
    response = requests.get(url, stream=True)

    # Check if the request was successful (status code 200).
    if response.status_code == 200:
        # Open the local file in binary write mode and write the downloaded content.
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)

        # Check if the saved file has a ".7z" extension.
        if save_path.lower().endswith(".7z"):
            # Open the 7z file and extract its contents to the same directory.
            with SevenZipFile(save_path, "r") as archive:
                archive.extractall(os.path.dirname(save_path))
            
            # Remove the original 7z file after extraction.
            os.remove(save_path)


def scrape_games(base_url, min_elo, save_path):
    """
    Scrapes chess game data from a website, filtering by minimum Elo rating,
    and downloads the corresponding PGN files.

    Args:
        base_url (str): The base URL of the website containing chess game data.
        min_elo (int): The minimum Elo rating for filtering games.
        save_path (str): The directory path to save the downloaded PGN files.
    """

    # Send an HTTP GET request to the base URL.
    response = requests.get(base_url)

    # Check if the request was successful (status code 200).
    if response.status_code == 200:

        # Parse HTML content using BeautifulSoup.
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all table rows with class "odd" or "even" containing game data.
        content = soup.find_all("tr", class_=["odd", "even"])
        
        # Initialize an empty list to store data links.
        data_links = []
        
        # Loop through each entry in the table.
        for entry in content:

            # Use a regular expression to extract Elo ratings from the entry.
            pattern = r"<b>(\d{1,4}|[2-9]\d{4,})</b>"
            numbers = re.findall(pattern, str(entry))
            numbers = [int(num) for num in numbers if int(num) > min_elo]
            
            # If there are Elo ratings greater than min_elo, extract the download link.
            if len(numbers) > 0:
                link = re.findall(r"href=\"games-by-engine-commented.+\.pgn\.7z", str(entry))
                data_links.append(link[0][6:])

        # Log the number of files to be downloaded.
        logging.info(f"Downloading {len(data_links)} files.")
        
        # Create the directory to save the downloaded files if it doesn't exist.
        os.makedirs(save_path, exist_ok=True)
        
        # Loop through each data link and download the corresponding PGN file.
        for link in tqdm(data_links, total=len(data_links), ncols=100, desc="Downloading"):
            url = urljoin(base_url, link)
            file_name = link.split('/')[-1]
            download_7z_file(url, os.path.join(save_path, file_name))
            

def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="data", help="The path to the PGN files.")
    parser.add_argument("--min_elo", type=int, default=2000, help="The minimum elo of the engine.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print logs.")
    return parser.parse_args()


def main():

    # Parse command line arguments.
    args = parse_args()

    # Configure logging if verbose mode is enabled.
    if args.verbose:
        get_logging_config()

    # Scrape the data.
    base_url = "https://computerchess.org.uk/ccrl/4040/"
    scrape_games(
        base_url=base_url, 
        min_elo=args.min_elo, 
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()