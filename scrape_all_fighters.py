#!/usr/bin/env python3
"""
Script to build a master list of UFC fighters and scrape data for all of them.
This script helps collect data on multiple fighters efficiently.
"""
# Used to modify the system path
import sys
# Handles file paths
from pathlib import Path
# Handles command line arguments
import argparse
# Displays progress bars
from tqdm import tqdm

# Tells the script where to find the src folder
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Imports the UFCScraper class from the scraper.py file
from data_collection.scraper import UFCScraper


def main():
    # Creates a parser for command line arguments
    parser = argparse.ArgumentParser(
        # Description of the script for when the user runs --help
        description="Build master fighter list and scrape UFC fighter data"
    )
    # Defines a cl flag called --build-list
    parser.add_argument(
        # Name of the flag
        '--build-list',
        # Action to store the value of the flag
        action='store_true',
        # Help message to display when the user runs --help
        help='Build master list of fighter URLs from events'
    )
    parser.add_argument(
        # Name of the flag
        '--num-events',
        # Means it must be followed by an integer
        type=int,
        # Default value of the flag
        default=50,
        # Help message to display when the user runs --help
        help='Number of recent events to scrape for fighter URLs (default: 50)'
    )
    parser.add_argument(
        # Name of the flag
        '--master-list',
        # Means it must be followed by a string
        type=str,
        # Default value of the flag
        default='data/raw/fighter_master_list.json',
        # Help message to display when the user runs --help
        help='Path to master list file (default: data/raw/fighter_master_list.json)'
    )
    parser.add_argument(
        # Name of the flag
        '--scrape',
        # If the flag is present, store True
        action='store_true',
        # Help message to display when the user runs --help
        help='Scrape data for all fighters in master list'
    )
    parser.add_argument(
        # Name of the flag
        '--skip-existing',
        # If the flag is present, store True
        action='store_true',
        # Default value of the flag
        default=True,
        # Help message
        help='Skip fighters that have already been scraped (default: True)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-scrape all fighters, including those already scraped (refresh data for new fights)'
    )
    parser.add_argument(
        # Name of the flag
        '--progress-file',
        # Means it must be followed by a string
        type=str,
        # Default value of the flag
        default='data/raw/scraping_progress.json',
        # Help message to display when the user runs --help
        help='Path to progress tracking file (default: data/raw/scraping_progress.json)'
    )
    parser.add_argument(
        # Name of the flag
        '--format',
        # Means it must be followed by a string
        type=str,
        # Choices that the user can choose from
        choices=['json', 'csv', 'both'],
        # Default value of the flag
        default='json',
        # Help message to display when the user runs --help
        help='Output format for saved data (default: json)'
    )
    
    # Reads the command line arguments and puts them into an object
    args = parser.parse_args()
    # Creates an instance of the UFCScraper class
    scraper = UFCScraper()
    
    # Checks if the user included the --build-list flag
    if args.build_list:
        # For users to see the progress of the script
        print("=" * 60)
        print("Building Master Fighter List")
        print("=" * 60)
        print(f"Number of events to process: {args.num_events}")
        print()
        
        # Calls build_fighter_master_list method to build the master list
        master_list = scraper.build_fighter_master_list(
            # Number of events to process
            num_events=args.num_events,
            # Path to save the master list
            save_path=args.master_list
        )
        
        # Print completion info
        print()
        print(f"✓ Master list created with {len(master_list)} unique fighter URLs")
        print(f"  Saved to: {args.master_list}")
        print()
    
    # Checks if the user included the --scrape flag
    if args.scrape:
        
        # For users to see the progress of the script
        print("=" * 60)
        print("Scraping Fighter Data")
        print("=" * 60)
        
        # Checks if the master list file exists
        if not Path(args.master_list).exists():
            # If the master list file does not exist, print an error message
            print(f"✗ Master list not found at: {args.master_list}")
            print("  Run with --build-list first to create the master list")
            return
        
        # Message to the user that the master list is being loaded
        print(f"Loading master list from: {args.master_list}")
        # Loads the master list from the file
        fighter_urls = scraper.load_fighter_master_list(args.master_list)
        
        # If no fighter URLs are found in the master list, print an error message and return
        if not fighter_urls:
            print("✗ No fighter URLs found in master list")
            return
        
        # Print the number of fighter URLs found in the master list
        print(f"Found {len(fighter_urls)} fighter URLs in master list")
        print()
        
        # Determines the save format based on the user's input
        save_format = 'json' if args.format in ['json', 'both'] else 'csv'
        # --no-skip-existing overrides --skip-existing to refresh all fighter data
        skip_existing = False if args.no_skip_existing else args.skip_existing

        # Prints the starting batch scrape info
        print(f"Starting batch scrape (format: {save_format}, skip existing: {skip_existing})")
        print()

        # Calls scrape_multiple_fighters method to scrape the data
        fighter_data = scraper.scrape_multiple_fighters(
            # List of fighter URLs to scrape
            fighter_urls=fighter_urls,
            # Save format
            save_format=save_format,
            # Skip existing fighters
            skip_existing=skip_existing,
            # Path to the progress file
            progress_file=args.progress_file
        )

        # If both formats requested, scrape again for CSV
        if args.format == 'both' and save_format == 'json':
            print()
            print("Scraping in CSV format...")
            scraper.scrape_multiple_fighters(
                fighter_urls=fighter_urls,
                save_format='csv',
                skip_existing=skip_existing,
                progress_file=args.progress_file
            )
        
        # Prints the completion info
        print()
        print("=" * 60)
        print("Scraping Complete!")
        print("=" * 60)
        print(f"Successfully scraped {len(fighter_data)} fighters")
        print(f"Data saved to: {scraper.raw_data_path}")
        print()
    
    # Checks if the user did not include the --build-list or --scrape flags
    if not args.build_list and not args.scrape:
        # Prints the help message
        parser.print_help()
        print()
        print("Examples:")
        print("  # Build master list from 50 recent events")
        print("  python scrape_all_fighters.py --build-list --num-events 50")
        print()
        print("  # Scrape all fighters in master list")
        print("  python scrape_all_fighters.py --scrape")
        print()
        print("  # Build list and scrape in one command")
        print("  python scrape_all_fighters.py --build-list --scrape")
        print()
        print("  # Refresh data (re-scrape all fighters, e.g. after new events)")
        print("  python scrape_all_fighters.py --scrape --no-skip-existing")


if __name__ == "__main__":
    main()

