#!/usr/bin/env python3
"""
Script to build a master list of UFC fighters and scrape data for all of them.
This script helps collect data on multiple fighters efficiently.
"""

import sys
from pathlib import Path
import argparse
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collection.scraper import UFCScraper


def main():
    parser = argparse.ArgumentParser(
        description="Build master fighter list and scrape UFC fighter data"
    )
    parser.add_argument(
        '--build-list',
        action='store_true',
        help='Build master list of fighter URLs from events and rankings'
    )
    parser.add_argument(
        '--num-events',
        type=int,
        default=50,
        help='Number of recent events to scrape for fighter URLs (default: 50)'
    )
    parser.add_argument(
        '--master-list',
        type=str,
        default='data/raw/fighter_master_list.json',
        help='Path to master list file (default: data/raw/fighter_master_list.json)'
    )
    parser.add_argument(
        '--scrape',
        action='store_true',
        help='Scrape data for all fighters in master list'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip fighters that have already been scraped (default: True)'
    )
    parser.add_argument(
        '--progress-file',
        type=str,
        default='data/raw/scraping_progress.json',
        help='Path to progress tracking file (default: data/raw/scraping_progress.json)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='json',
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
            include_rankings=True,
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

