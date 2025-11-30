#!/usr/bin/env python3
"""Test script for the UFC scraper."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collection.scraper import UFCScraper

def test_scraper(fighter_url=None):
    """Test the scraper with a real fighter URL.
    
    Args:
        fighter_url: Optional fighter URL to test with. If not provided,
                    will use an example URL or prompt for one.
    """
    
    print("=" * 60)
    print("UFC Scraper Test Script")
    print("=" * 60)
    print()
    
    # Initialize scraper
    print("Initializing scraper...")
    try:
        scraper = UFCScraper()
        print("✓ Scraper initialized successfully")
        print(f"  Base URL: {scraper.base_url}")
        print(f"  Rate limit delay: {scraper.rate_limit_delay}s")
        print(f"  Raw data path: {scraper.raw_data_path}")
    except Exception as e:
        print(f"✗ Failed to initialize scraper: {e}")
        return
    
    print()
    
    # Get fighter URL
    if not fighter_url:
        print("To test the scraper, you need a fighter URL from ufcstats.com")
        print("Example format: http://ufcstats.com/fighter-details/XXXXXXXX")
        print()
        print("You can find fighter URLs by:")
        print("  1. Visit http://ufcstats.com")
        print("  2. Navigate to any fighter's page")
        print("  3. Copy the URL from your browser")
        print()
        
        fighter_url = input("Enter a fighter URL (or press Enter to skip test): ").strip()
        
        if not fighter_url:
            print("\nSkipping scraper test. Exiting.")
            return
    
    print()
    print("-" * 60)
    print(f"Testing scraper with URL: {fighter_url}")
    print("-" * 60)
    print()
    
    # Scrape fighter data
    try:
        print("Scraping fighter data...")
        fighter_data = scraper.scrape_fighter(fighter_url)
        
        if fighter_data:
            print("✓ Scraping successful!")
            print()
            
            # Display basic info
            stats = fighter_data.get('stats', {})
            name = stats.get('name', 'Unknown')
            record = stats.get('record', 'N/A')
            num_fights = len(fighter_data.get('fight_history', []))
            
            print(f"Fighter Name: {name}")
            print(f"Record: {record}")
            print(f"Number of fights scraped: {num_fights}")
            print()
            
            # Display detailed stats
            if stats:
                print("Fighter Statistics:")
                print("-" * 40)
                for key, value in sorted(stats.items()):
                    if value:  # Only show non-empty values
                        print(f"  {key}: {value}")
                print()
            
            # Display fight history sample
            fight_history = fighter_data.get('fight_history', [])
            if fight_history:
                print(f"Fight History (showing first 5 of {len(fight_history)}):")
                print("-" * 40)
                for i, fight in enumerate(fight_history[:5], 1):
                    opponent = fight.get('opponent', 'Unknown')
                    result = fight.get('result', 'N/A')
                    method = fight.get('method', 'N/A')
                    date = fight.get('date', 'N/A')
                    print(f"  {i}. vs {opponent}: {result} ({method}) - {date}")
                if len(fight_history) > 5:
                    print(f"  ... and {len(fight_history) - 5} more fights")
                print()
            
            # Save the data
            print("Saving fighter data...")
            try:
                scraper.save_fighter_data(fighter_data, format='json')
                print("✓ Data saved as JSON to data/raw/")
                
                scraper.save_fighter_data(fighter_data, format='csv')
                print("✓ Data saved as CSV to data/raw/")
            except Exception as e:
                print(f"✗ Error saving data: {e}")
            
        else:
            print("✗ Scraping failed - no data returned")
            print("\nPossible reasons:")
            print("  - Invalid or inaccessible URL")
            print("  - Network connectivity issues")
            print("  - Website structure may have changed")
            print("  - Rate limiting or blocking")
            
    except Exception as e:
        print(f"✗ Error during scraping: {e}")
        import traceback
        print("\nFull error details:")
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Test completed")
    print("=" * 60)


def test_with_example_urls():
    """Test scraper with some example fighter URLs (you'll need to update these)."""
    
    print("Testing with example URLs...")
    print("(Note: You'll need to replace these with actual fighter URLs)")
    print()
    
    # These are placeholder URLs - replace with real ones from ufcstats.com
    example_urls = [
        # "http://ufcstats.com/fighter-details/EXAMPLE1",
        # "http://ufcstats.com/fighter-details/EXAMPLE2",
    ]
    
    if not example_urls:
        print("No example URLs configured. Use test_scraper() with a specific URL.")
        return
    
    scraper = UFCScraper()
    
    for url in example_urls:
        print(f"\nTesting: {url}")
        fighter_data = scraper.scrape_fighter(url)
        if fighter_data:
            scraper.save_fighter_data(fighter_data)
            print(f"✓ Successfully scraped {fighter_data.get('stats', {}).get('name', 'Unknown')}")
        else:
            print("✗ Failed to scrape")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the UFC scraper")
    parser.add_argument(
        "--url",
        type=str,
        help="Fighter URL to test with (from ufcstats.com)"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run tests with example URLs"
    )
    
    args = parser.parse_args()
    
    if args.examples:
        test_with_example_urls()
    else:
        test_scraper(fighter_url=args.url)

