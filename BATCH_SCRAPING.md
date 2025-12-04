# Batch Scraping Guide

This guide explains how to scrape data for multiple UFC fighters at once.

## Quick Start

### Step 1: Build Master Fighter List

First, build a master list of fighter URLs from events:

```bash
python scrape_all_fighters.py --build-list --num-events 50
```

This will:
- Scrape fighter URLs from the 50 most recent UFC events
- Save the master list to `data/raw/fighter_master_list.json`

### Step 2: Scrape All Fighters

Once you have the master list, scrape data for all fighters:

```bash
python scrape_all_fighters.py --scrape
```

This will:
- Load the master list
- Skip fighters that have already been scraped (to resume interrupted jobs)
- Save progress to `data/raw/scraping_progress.json`
- Save fighter data to `data/raw/` directory

### Do Both in One Command

```bash
python scrape_all_fighters.py --build-list --scrape --num-events 50
```

## Command Options

### Building Master List

- `--build-list`: Build master list of fighter URLs
- `--num-events N`: Number of recent events to process (default: 50)
- `--master-list PATH`: Where to save the master list (default: `data/raw/fighter_master_list.json`)

### Scraping Fighters

- `--scrape`: Scrape data for all fighters in master list
- `--skip-existing`: Skip fighters already scraped (default: True)
- `--format FORMAT`: Output format - `json`, `csv`, or `both` (default: `json`)
- `--progress-file PATH`: Path to progress tracking file (default: `data/raw/scraping_progress.json`)

## Examples

### Build list from last 100 events
```bash
python scrape_all_fighters.py --build-list --num-events 100
```

### Scrape and save as both JSON and CSV
```bash
python scrape_all_fighters.py --scrape --format both
```

### Resume interrupted scraping
The scraper automatically resumes from where it left off if you run it again:
```bash
python scrape_all_fighters.py --scrape  # Automatically skips already scraped fighters
```

## Using the Scraper Programmatically

### Build Master List

```python
from src.data_collection.scraper import UFCScraper

scraper = UFCScraper()

# Build master list from 50 events
master_list = scraper.build_fighter_master_list(
    num_events=50,
    save_path='data/raw/fighter_master_list.json'
)

print(f"Found {len(master_list)} unique fighters")
```

### Scrape Multiple Fighters

```python
from src.data_collection.scraper import UFCScraper

scraper = UFCScraper()

# Load master list
fighter_urls = scraper.load_fighter_master_list('data/raw/fighter_master_list.json')

# Scrape all fighters (automatically skips already scraped ones)
fighter_data = scraper.scrape_multiple_fighters(
    fighter_urls=fighter_urls,
    save_format='json',
    skip_existing=True,
    progress_file='data/raw/scraping_progress.json'
)
```

### Get Fighters from Events Only

```python
# Get fighters from last 10 events
fighter_urls = scraper.get_fighters_from_events(num_events=10)
```


## Important Notes

1. **Rate Limiting**: The scraper includes built-in rate limiting (2 seconds between requests by default) to avoid being blocked.

2. **Resume Capability**: If scraping is interrupted, you can resume by running the same command again. It will automatically skip fighters that have already been scraped.

3. **Progress Tracking**: Progress is saved periodically (every 10 fighters) so you don't lose progress if the script crashes.

4. **Time Required**: Scraping hundreds of fighters can take a while:
   - ~2 seconds per fighter (rate limiting)
   - 100 fighters = ~3-4 minutes
   - 500 fighters = ~15-20 minutes
   - 1000+ fighters = 30+ minutes

5. **Data Storage**: All scraped data is saved to `data/raw/` directory:
   - JSON format: `FighterName_data.json`
   - CSV format: `FighterName_stats.csv` and `FighterName_fights.csv`

## Troubleshooting

**Problem**: Master list is empty
- **Solution**: Check if ufcstats.com structure has changed. The event page structure may need to be updated in the scraper code.

**Problem**: Scraping stops partway through
- **Solution**: Just run the command again. It will automatically resume from where it left off.

**Problem**: Getting rate-limited or blocked
- **Solution**: Increase the rate limit delay in `config/config.yaml` (scraping.rate_limit_delay)

