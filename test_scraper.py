#!/usr/bin/env python3
"""Test script for the UFC scraper."""
# Gives the script access to the system path
import sys
# Lets the script modify the system path
from pathlib import Path

# Add src to import path 
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Imports the UFCScraper class from the scraper.py file
from data_collection.scraper import UFCScraper

# Defines a function to test the scraper
def test_scraper(fighter_url=None):
    """Test the scraper with a real fighter URL.
    
    Args:
        fighter_url: Optional fighter URL to test with. If not provided,
                    will use an example URL or prompt for one.
    """
    
    # Prints the title of the script
    print("=" * 60)
    print("UFC Scraper Test Script")
    print("=" * 60)
    print()
    
    # Prints the initializing scraper message
    print("Initializing scraper...")
    # Tries to initialize the scraper
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
    
    # If no fighter URL is provided, prompt the user for one
    if not fighter_url:
        print("To test the scraper, you need a fighter URL from ufcstats.com")
        print("Example format: http://ufcstats.com/fighter-details/XXXXXXXX")
        print()
        print("You can find fighter URLs by:")
        print("  1. Visit http://ufcstats.com")
        print("  2. Navigate to any fighter's page")
        print("  3. Copy the URL from your browser")
        print()
        
        # Gets the fighter URL from the user
        fighter_url = input("Enter a fighter URL (or press Enter to skip test): ").strip()
        
        # If the user does not enter a fighter URL, print a message and return
        if not fighter_url:
            print("\nSkipping scraper test. Exiting.")
            return
    
    print()
    # Prints the fighter URL being tested
    print("-" * 60)
    print(f"Testing scraper with URL: {fighter_url}")
    print("-" * 60)
    print()
    
    # Scrape fighter data
    try:
        print("Scraping fighter data...")
        # Scrapes the fighter data
        fighter_data = scraper.scrape_fighter(fighter_url)
        
        # If the fighter data is found, print a success message
        if fighter_data:
            print("✓ Scraping successful!")
            print()
            
            # Gets the stats of the fighter
            stats = fighter_data.get('stats', {})
            # Gets the name of the fighter
            name = stats.get('name', 'Unknown')
            # Gets the record of the fighter
            record = stats.get('record', 'N/A')
            # Gets the number of fights the fighter has been in
            num_fights = len(fighter_data.get('fight_history', []))

            # Prints the fighter name, record, and number of fights
            print(f"Fighter Name: {name}")
            print(f"Record: {record}")
            print(f"Number of fights scraped: {num_fights}")
            print()
            
            # If stats exist, print the stats
            if stats:
                print("Fighter Statistics:")
                print("-" * 40)
                # Goes through each stat and prints it
                for key, value in sorted(stats.items()):
                    # Only shows non-empty values
                    if value:  # Only show non-empty values
                        print(f"  {key}: {value}")
                print()
            
            # Gets fighter history from fighter_data
            fight_history = fighter_data.get('fight_history', [])
            # Checks if the fighter has a fight history
            if fight_history:
                # Prints last 5 fights of the fighter
                print(f"Fight History (showing first 5 of {len(fight_history)}):")
                print("-" * 40)
                # Prints the last 5 fights of the fighter and counts the number of fights that are not in the last 5
                for i, fight in enumerate(fight_history[:5], 1):
                    opponent = fight.get('opponent', 'Unknown')
                    result = fight.get('result', 'N/A')
                    method = fight.get('method', 'N/A')
                    date = fight.get('date', 'N/A')
                    # Prints the fight information
                    print(f"  {i}. vs {opponent}: {result} ({method}) - {date}")
                if len(fight_history) > 5:
                    print(f"  ... and {len(fight_history) - 5} more fights")
                print()
            
            # Save the data
            print("Saving fighter data...")
            try:
                # Saves the fighter data as JSON
                scraper.save_fighter_data(fighter_data, format='json')
                print("✓ Data saved as JSON to data/raw/")
                # Saves the fighter data as CSV
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

# Only runs if the script is being run directly
if __name__ == "__main__":
    # Imports the argparse module (used to parse command line arguments)
    import argparse
    
    # Creats an ArgumentParser object
    parser = argparse.ArgumentParser(description="Test the UFC scraper")
    # Adds a command line argument for the fighter URL
    parser.add_argument(
        # Name of the argument
        "--url",
        # Type of the argument
        type=str,
        # Help message for the argument
        help="Fighter URL to test with (from ufcstats.com)"
    )
    # Adds a command line argument for the example URLs
    parser.add_argument(
        # Name of the argument
        "--examples",
        # Type of the argument
        action="store_true", # True or False
        # Help message for the argument
        help="Run tests with example URLs"
    )
    
    # Parses the command line arguments
    args = parser.parse_args()
    
    # If the user wants to test with example URLs, run the test_with_example_urls function
    if args.examples:
        test_with_example_urls()
    # If the user wants to test with a specific fighter URL, run the test_scraper function
    else:
        test_scraper(fighter_url=args.url)

