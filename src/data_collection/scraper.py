"""
Web scraper for collecting UFC fighter statistics and fight history.
Scrapes data from ufcstats.com including fighter profiles, fight records, and detailed statistics.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
from urllib.parse import urljoin, urlparse
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UFCScraper:
    """Scraper for UFC fighter statistics and fight data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize scraper with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.base_url = self.config.get('scraping', {}).get('base_url', 'http://ufcstats.com')
        self.rate_limit_delay = self.config.get('scraping', {}).get('rate_limit_delay', 2)
        self.timeout = self.config.get('scraping', {}).get('timeout', 30)
        self.retry_attempts = self.config.get('scraping', {}).get('retry_attempts', 3)
        self.user_agent = self.config.get('scraping', {}).get('user_agent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        self.raw_data_path = Path(self.config.get('paths', {}).get('raw_data', 'data/raw'))
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and rate limiting.
        
        Args:
            url: URL to request
            
        Returns:
            Response object or None if request failed
        """
        # Try the request multiple times (based on retry_attempts)
        for attempt in range(self.retry_attempts):
            try:
                # To avoid being blocked, we need to wait for the rate limit delay
                time.sleep(self.rate_limit_delay)
                # Send the request (includes session, agent, and timeout)
                response = self.session.get(url, timeout=self.timeout)
                # Raise an exception if the request failed
                response.raise_for_status()
                # Return the response
                return response
            # This exception is raised if the request failed
            except requests.exceptions.RequestException as e:
                # Log the error
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                # If this is the last attempt, log the error and return None
                if attempt == self.retry_attempts - 1:
                    logger.error(f"Failed to fetch {url} after {self.retry_attempts} attempts")
                    return None
                # Otherwise, wait for the rate limit delay multiplied by the attempt number (give server some time to recover)
                time.sleep(self.rate_limit_delay * (attempt + 1))
        return None
    # Takes in a BeautifulSoup object of the fighter page and returns a dictionary of the fighter's statistics
    def _parse_fighter_stats(self, soup: BeautifulSoup) -> Dict:
        """Parse fighter statistics from fighter detail page.
        
        Args:
            soup: BeautifulSoup object of fighter page
            
        Returns:
            Dictionary containing fighter statistics
        """
        # Initialize an empty dictionary to hold parsed statistics
        stats = {}
        # Try to parse the statistics
        try:
            # Find the first span with the class 'b-content__title-highlight' and extract the text
            name_elem = soup.find('span', class_='b-content__title-highlight')
            # If the name element is found, add the name to the statistics dictionary (get_text(strip=True) removes whitespace)
            if name_elem:
                stats['name'] = name_elem.get_text(strip=True)
            
            # Find the span with the class 'b-content__title-record' and extract the text
            record_elem = soup.find('span', class_='b-content__title-record')
            # If the record element is found, add the record to the statistics dictionary (get_text(strip=True) removes whitespace)
            if record_elem:
                record_text = record_elem.get_text(strip=True)
                # Remove the 'Record: ' prefix from the record text
                stats['record'] = record_text.replace('Record: ', '')
                # Parse the record text into wins, losses, and draws
                # Parse W-L-D
                if '-' in stats['record']:
                    # Split the record text into wins, losses, and draws
                    parts = stats['record'].split('-')
                    # If there are at least 2 parts, add the wins, losses, and draws to the statistics dictionary
                    if len(parts) >= 2:
                        # Add the wins to the statistics dictionary
                        stats['wins'] = int(parts[0].strip())
                        stats['losses'] = int(parts[1].strip())
                        stats['draws'] = int(parts[2].strip()) if len(parts) > 2 else 0
            
            # Finds the FIRST div with the class 'b-list__info-box' and extracts the text of the list items
            stats_table = soup.find('div', class_='b-list__info-box')
            # If the stats table is found, extract the text of the list items
            if stats_table:
                items = stats_table.find_all('li', class_='b-list__box-list-item')
                for item in items:
                    text = item.get_text(strip=True)
                    if 'Height:' in text:
                        stats['height'] = text.replace('Height:', '').strip()
                    elif 'Weight:' in text:
                        stats['weight'] = text.replace('Weight:', '').strip()
                    elif 'Reach:' in text:
                        stats['reach'] = text.replace('Reach:', '').strip()
                    elif 'STANCE:' in text:
                        stats['stance'] = text.replace('STANCE:', '').strip()
                    elif 'DOB:' in text:
                        stats['dob'] = text.replace('DOB:', '').strip()
            
            # Extract career statistics
            career_stats = soup.find_all('div', class_='b-list__info-box-left')
            for stat_box in career_stats:
                items = stat_box.find_all('li', class_='b-list__box-list-item')
                for item in items:
                    text = item.get_text(strip=True)
                    if 'SLpM:' in text:
                        stats['strikes_landed_per_min'] = self._extract_number(text)
                    elif 'Str. Acc.:' in text:
                        stats['striking_accuracy'] = self._extract_number(text.replace('%', ''))
                    elif 'SApM:' in text:
                        stats['strikes_absorbed_per_min'] = self._extract_number(text)
                    elif 'Str. Def:' in text:
                        stats['striking_defense'] = self._extract_number(text.replace('%', ''))
                    elif 'TD Avg.:' in text:
                        stats['takedown_average'] = self._extract_number(text)
                    elif 'TD Acc.:' in text:
                        stats['takedown_accuracy'] = self._extract_number(text.replace('%', ''))
                    elif 'TD Def.:' in text:
                        stats['takedown_defense'] = self._extract_number(text.replace('%', ''))
                    elif 'Sub. Avg.:' in text:
                        stats['submission_average'] = self._extract_number(text)
            
        except Exception as e:
            logger.error(f"Error parsing fighter stats: {e}")
        
        return stats
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text string.
        
        Args:
            text: Text containing number
            
        Returns:
            Numeric value or None
        """
        try:
            # Remove common non-numeric characters except decimal point and minus
            cleaned = ''.join(c for c in text if c.isdigit() or c == '.' or c == '-')
            if cleaned:
                return float(cleaned)
        except (ValueError, AttributeError):
            pass
        return None
    
    def _parse_fight_history(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse fight history from fighter page.
        
        Args:
            soup: BeautifulSoup object of fighter page
            
        Returns:
            List of dictionaries containing fight information
        """
        fights = []
        
        try:
            # Find fight history table
            fight_table = soup.find('table', class_='b-fight-details__table')
            if not fight_table:
                return fights
            
            rows = fight_table.find('tbody').find_all('tr', class_='b-fight-details__table-row')
            
            for row in rows:
                fight = {}
                
                # Get all table cells in order
                cells = row.find_all('td', class_='b-fight-details__table-col')
                
                # Extract opponent name (usually in first cell with links)
                opponent_links = row.find_all('a', class_='b-link')
                if len(opponent_links) >= 2:
                    fight['opponent'] = opponent_links[1].get_text(strip=True)
                
                # Extract result (Win/Loss/Draw/NC) - first p tag with result class
                result_elems = row.find_all('p', class_='b-fight-details__table-text')
                if result_elems:
                    # First element is usually the result (Win/Loss)
                    result_text = result_elems[0].get_text(strip=True)
                    fight['result'] = result_text.lower() if result_text else 'N/A'
                
                # Parse each cell to extract method, round, date, etc.
                for cell in cells:
                    cell_text = cell.get_text(strip=True)
                    
                    # Look for date in the cell (dates contain month abbreviations)
                    month_abbrevs = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    if any(month in cell_text for month in month_abbrevs):
                        # Date found - extract it from the p tags in this cell
                        date_p_tags = cell.find_all('p', class_='b-fight-details__table-text')
                        for p_tag in date_p_tags:
                            p_text = p_tag.get_text(strip=True)
                            if any(month in p_text for month in month_abbrevs):
                                fight['date'] = p_text
                                break
                    
                    # Look for method in cell (method cells contain method types)
                    p_tags = cell.find_all('p', class_='b-fight-details__table-text')
                    if len(p_tags) >= 2:
                        # Check if this looks like a method cell (has method abbreviation and description)
                        first_p = p_tags[0].get_text(strip=True).upper()
                        second_p = p_tags[1].get_text(strip=True)
                        
                        # Method indicators
                        if first_p in ['KO/TKO', 'SUB', 'DEC', 'DQ', 'NC']:
                            if first_p == 'SUB':
                                fight['method'] = f"Submission ({second_p})" if second_p else "Submission"
                            elif first_p == 'DEC':
                                fight['method'] = f"Decision ({second_p})" if second_p else "Decision"
                            elif first_p in ['KO/TKO']:
                                fight['method'] = f"{first_p} ({second_p})" if second_p else first_p
                            else:
                                fight['method'] = first_p if second_p == '' else f"{first_p} ({second_p})"
                    
                    # Look for round information
                    if 'Round' in cell_text or 'Rd' in cell_text:
                        round_text = cell_text
                        round_num = self._extract_number(round_text)
                        if round_num:
                            fight['round'] = int(round_num)
                
                # Set defaults if not found
                if 'date' not in fight:
                    fight['date'] = 'N/A'
                if 'method' not in fight:
                    fight['method'] = 'N/A'
                
                fights.append(fight)
                
        except Exception as e:
            logger.error(f"Error parsing fight history: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return fights
    
    def get_fighter_page_url(self, fighter_url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a fighter's detail page.
        
        Args:
            fighter_url: URL to fighter's page
            
        Returns:
            BeautifulSoup object or None if failed
        """
        response = self._make_request(fighter_url)
        if response:
            return BeautifulSoup(response.content, 'lxml')
        return None
    
    def scrape_fighter(self, fighter_url: str) -> Optional[Dict]:
        """Scrape complete fighter data including stats and fight history.
        
        Args:
            fighter_url: URL to fighter's page
            
        Returns:
            Dictionary containing all fighter data or None if failed
        """
        logger.info(f"Scraping fighter: {fighter_url}")
        
        soup = self.get_fighter_page_url(fighter_url)
        if not soup:
            return None
        
        fighter_data = {
            'url': fighter_url,
            'stats': self._parse_fighter_stats(soup),
            'fight_history': self._parse_fight_history(soup)
        }
        
        return fighter_data
    
    def get_fighters_list(self, event_url: Optional[str] = None) -> List[str]:
        """Get list of fighter URLs from events or fighter list page.
        
        Args:
            event_url: Optional URL to specific event page
            
        Returns:
            List of fighter URLs
        """
        fighter_urls = []
        
        # For now, return empty list - this would need to be implemented
        # based on the structure of ufcstats.com fighter listing pages
        # This could scrape from events, fighter rankings, or fighter list pages
        
        return fighter_urls
    
    def save_fighter_data(self, fighter_data: Dict, format: str = 'json') -> None:
        """Save fighter data to file.
        
        Args:
            fighter_data: Dictionary containing fighter data
            format: Output format ('json' or 'csv')
        """
        if not fighter_data or 'stats' not in fighter_data:
            logger.warning("No valid fighter data to save")
            return
        
        fighter_name = fighter_data.get('stats', {}).get('name', 'unknown')
        safe_name = "".join(c for c in fighter_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if format == 'json':
            filename = self.raw_data_path / f"{safe_name}_data.json"
            with open(filename, 'w') as f:
                json.dump(fighter_data, f, indent=2)
        elif format == 'csv':
            # Save stats as CSV
            stats_filename = self.raw_data_path / f"{safe_name}_stats.csv"
            if fighter_data.get('stats'):
                with open(stats_filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fighter_data['stats'].keys())
                    writer.writeheader()
                    writer.writerow(fighter_data['stats'])
            
            # Save fight history as separate CSV
            if fighter_data.get('fight_history'):
                history_filename = self.raw_data_path / f"{safe_name}_fights.csv"
                with open(history_filename, 'w', newline='') as f:
                    if fighter_data['fight_history']:
                        writer = csv.DictWriter(f, fieldnames=fighter_data['fight_history'][0].keys())
                        writer.writeheader()
                        writer.writerows(fighter_data['fight_history'])
        
        logger.info(f"Saved fighter data for {fighter_name} to {filename if format == 'json' else stats_filename}")
    
    def scrape_multiple_fighters(self, fighter_urls: List[str], save_format: str = 'json') -> List[Dict]:
        """Scrape data for multiple fighters.
        
        Args:
            fighter_urls: List of fighter URLs to scrape
            save_format: Format to save data ('json' or 'csv')
            
        Returns:
            List of fighter data dictionaries
        """
        all_fighter_data = []
        
        for i, url in enumerate(fighter_urls, 1):
            logger.info(f"Scraping fighter {i}/{len(fighter_urls)}")
            fighter_data = self.scrape_fighter(url)
            
            if fighter_data:
                all_fighter_data.append(fighter_data)
                self.save_fighter_data(fighter_data, format=save_format)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        return all_fighter_data


def main():
    """Example usage of the UFC scraper."""
    scraper = UFCScraper()
    
    # Example: Scrape a specific fighter
    # fighter_url = "http://ufcstats.com/fighter-details/12345"
    # fighter_data = scraper.scrape_fighter(fighter_url)
    # if fighter_data:
    #     scraper.save_fighter_data(fighter_data)
    
    logger.info("UFC Scraper initialized. Use scraper.scrape_fighter(url) to scrape fighter data.")


if __name__ == "__main__":
    main()

