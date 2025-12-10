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
    
    args = parser.parse_args()
    
    scraper = UFCScraper()
    
    # Build master list if requested
    if args.build_list:
        print("=" * 60)
        print("Building Master Fighter List")
        print("=" * 60)
        print(f"Number of events to process: {args.num_events}")
        print()
        
        master_list = scraper.build_fighter_master_list(
            num_events=args.num_events,
            save_path=args.master_list
        )
        
        print()
        print(f"✓ Master list created with {len(master_list)} unique fighter URLs")
        print(f"  Saved to: {args.master_list}")
        print()
    
    # Scrape fighters if requested
    if args.scrape:
        print("=" * 60)
        print("Scraping Fighter Data")
        print("=" * 60)
        
        # Load master list
        if not Path(args.master_list).exists():
            print(f"✗ Master list not found at: {args.master_list}")
            print("  Run with --build-list first to create the master list")
            return
        
        print(f"Loading master list from: {args.master_list}")
        fighter_urls = scraper.load_fighter_master_list(args.master_list)
        
        if not fighter_urls:
            print("✗ No fighter URLs found in master list")
            return
        
        print(f"Found {len(fighter_urls)} fighter URLs in master list")
        print()
        
        # Determine save format
        save_format = 'json' if args.format in ['json', 'both'] else 'csv'
        
        # Scrape all fighters
        print(f"Starting batch scrape (format: {save_format}, skip existing: {args.skip_existing})")
        print()
        
        fighter_data = scraper.scrape_multiple_fighters(
            fighter_urls=fighter_urls,
            save_format=save_format,
            skip_existing=args.skip_existing,
            progress_file=args.progress_file
        )
        
        # If both formats requested, scrape again for CSV
        if args.format == 'both' and save_format == 'json':
            print()
            print("Scraping in CSV format...")
            scraper.scrape_multiple_fighters(
                fighter_urls=fighter_urls,
                save_format='csv',
                skip_existing=True,  # Only scrape new ones
                progress_file=args.progress_file
            )
        
        print()
        print("=" * 60)
        print("Scraping Complete!")
        print("=" * 60)
        print(f"Successfully scraped {len(fighter_data)} fighters")
        print(f"Data saved to: {scraper.raw_data_path}")
        print()
    
    if not args.build_list and not args.scrape:
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


if __name__ == "__main__":
    main()

