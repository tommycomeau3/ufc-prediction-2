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
                # Find all list items with the class 'b-list__box-list-item'
                items = stats_table.find_all('li', class_='b-list__box-list-item')
                # Go through each soup object and extract the text
                for item in items:
                   # Get the text of the list item and remove whitespace
                    text = item.get_text(strip=True)
                    # If the text contains 'Height:', add the height to the statistics dictionary
                    if 'Height:' in text:
                        stats['height'] = text.replace('Height:', '').strip()
                    # If the text contains 'Weight:', add the weight to the statistics dictionary
                    elif 'Weight:' in text:
                        stats['weight'] = text.replace('Weight:', '').strip()
                    # If the text contains 'Reach:', add the reach to the statistics dictionary
                    elif 'Reach:' in text:
                        stats['reach'] = text.replace('Reach:', '').strip()
                    # If the text contains 'STANCE:', add the stance to the statistics dictionary
                    elif 'STANCE:' in text:
                        stats['stance'] = text.replace('STANCE:', '').strip()
                    # If the text contains 'DOB:', add the date of birth to the statistics dictionary
                    elif 'DOB:' in text:
                        stats['dob'] = text.replace('DOB:', '').strip()
            
            # Finds all divs with the class 'b-list__info-box-left' and extracts the text of the list items
            career_stats = soup.find_all('div', class_='b-list__info-box-left')
            # Go through each stat box and extract the text of the list items
            for stat_box in career_stats:
                
                items = stat_box.find_all('li', class_='b-list__box-list-item')
                for item in items:
                    # Get the text of the list item and remove whitespace
                    text = item.get_text(strip=True)
                    # If the text contains 'SLpM:', add the strikes landed per minute to the statistics dictionary
                    if 'SLpM:' in text:
                        stats['strikes_landed_per_min'] = self._extract_number(text)
                    # If the text contains 'Str. Acc.:', add the striking accuracy to the statistics dictionary
                    elif 'Str. Acc.:' in text:
                        stats['striking_accuracy'] = self._extract_number(text.replace('%', ''))
                    # If the text contains 'SApM:', add the strikes absorbed per minute to the statistics dictionary
                    elif 'SApM:' in text:
                        stats['strikes_absorbed_per_min'] = self._extract_number(text)
                    # If the text contains 'Str. Def.:', add the striking defense to the statistics dictionary
                    elif 'Str. Def:' in text:
                        stats['striking_defense'] = self._extract_number(text.replace('%', ''))
                    # If the text contains 'TD Avg.:', add the takedown average to the statistics dictionary
                    elif 'TD Avg.:' in text:
                        stats['takedown_average'] = self._extract_number(text)
                    # If the text contains 'TD Acc.:', add the takedown accuracy to the statistics dictionary
                    elif 'TD Acc.:' in text:
                        stats['takedown_accuracy'] = self._extract_number(text.replace('%', ''))
                    # If the text contains 'TD Def.:', add the takedown defense to the statistics dictionary
                    elif 'TD Def.:' in text:
                        stats['takedown_defense'] = self._extract_number(text.replace('%', ''))
                    # If the text contains 'Sub. Avg.:', add the submission average to the statistics dictionary
                    elif 'Sub. Avg.:' in text:
                        stats['submission_average'] = self._extract_number(text)
        # If an error occurs, log the error
        except Exception as e:
            logger.error(f"Error parsing fighter stats: {e}")
        # Return the statistics dictionary
        return stats
    
    # Takes a string and returns a float if the string is a number otherwise returns None
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
            # If the cleaned string is not empty it returns a float of the string
            if cleaned:
                return float(cleaned)
        # If something goes wrong, return None
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
        # Initialize an empty list to hold the fight information
        fights = []
        
        try:
            # Find the first table with class 'b-fight-details__table'
            fight_table = soup.find('table', class_='b-fight-details__table')
            # Return an empty list if fight table does not exist
            if not fight_table:
                return fights
            # Finds the tbody of the fight table and finds all trs with the class 'b-fight-details__table-row'
            rows = fight_table.find('tbody').find_all('tr', class_='b-fight-details__table-row')
            # Go through each row in rows and extract the fight information
            for row in rows:
                # Initialize an empty dictionary to hold the fight information
                fight = {}
                
                # Find all td with the class 'b-fight-details__table-col'
                cells = row.find_all('td', class_='b-fight-details__table-col')
                
                # Finds all a with the class 'b-link' and extracts the text
                opponent_links = row.find_all('a', class_='b-link')
                # If there are at least 2 opponent links, add the second link's text to the fight dictionary
                if len(opponent_links) >= 2:
                    fight['opponent'] = opponent_links[1].get_text(strip=True)
                
                # Extract result (Win/Loss/Draw/NC) - first p tag with result class
                result_elems = row.find_all('p', class_='b-fight-details__table-text')
                # If the result elements are found, add the result to the fight dictionary
                if result_elems:
                    # First element is usually the result (Win/Loss)
                    result_text = result_elems[0].get_text(strip=True)
                    # Add the result to the fight dictionary and convert to lowercase
                    fight['result'] = result_text.lower() if result_text else 'N/A'
                
                # Parse each cell to extract method, round, date, etc.
                for cell in cells:
                    # Extract the text of the cell and remove whitespace
                    cell_text = cell.get_text(strip=True)
                    
                    # Look for date in the cell (dates contain month abbreviations)
                    month_abbrevs = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    # Checks if any of the month abbreviations are in the cell text
                    if any(month in cell_text for month in month_abbrevs):
                        # Find all p tags with the class 'b-fight-details__table-text'
                        date_p_tags = cell.find_all('p', class_='b-fight-details__table-text')
                        
                        # Go through each p tag and extract the text
                        for p_tag in date_p_tags:
                            # Get the text of the p tag and remove whitespace
                            p_text = p_tag.get_text(strip=True)
                            # Checks if any of the month abbreviations are in the p text
                            if any(month in p_text for month in month_abbrevs):
                                # Add the date to the fight dictionary
                                fight['date'] = p_text
                                break
                    
                    # Find all p tags with the class 'b-fight-details__table-text'
                    p_tags = cell.find_all('p', class_='b-fight-details__table-text')
                    # Make sure there are at least 2 p tags
                    if len(p_tags) >= 2:
                        # Strips whitespace and converts to uppercase
                        first_p = p_tags[0].get_text(strip=True).upper()
                        # Strips whitespace
                        second_p = p_tags[1].get_text(strip=True)
                        
                        # Check for method
                        if first_p in ['KO/TKO', 'SUB', 'DEC', 'DQ', 'NC']:
                            # If the first p tag is a submission, add the method to the fight dictionary
                            if first_p == 'SUB':
                                fight['method'] = f"Submission ({second_p})" if second_p else "Submission"
                            # If the first p tag is a decision, add the method to the fight dictionary
                            elif first_p == 'DEC':
                                fight['method'] = f"Decision ({second_p})" if second_p else "Decision"
                            # If the first p tag is a KO/TKO, add the method to the fight dictionary
                            elif first_p in ['KO/TKO']:
                                fight['method'] = f"{first_p} ({second_p})" if second_p else first_p
                            # If the first p tag is a DQ, add the method to the fight dictionary
                            else:
                                fight['method'] = first_p if second_p == '' else f"{first_p} ({second_p})"
                    
                    # Checks if the cell text contains 'Round' or 'Rd'
                    if 'Round' in cell_text or 'Rd' in cell_text:
                        # Save the cell text to the round text
                        round_text = cell_text
                        # Extract the number from the round text
                        round_num = self._extract_number(round_text)
                        # If the round number is found, add it to the fight dictionary
                        if round_num:
                            fight['round'] = int(round_num)
                
                # If the date is not found, add 'N/A' to the fight dictionary
                if 'date' not in fight:
                    fight['date'] = 'N/A'
                # If the method is not found, add 'N/A' to the fight dictionary
                if 'method' not in fight:
                    fight['method'] = 'N/A'
                # Add the fight dictionary to the list of fights
                fights.append(fight)
        # If an error occurs, log the error and print the traceback
        except Exception as e:
            logger.error(f"Error parsing fight history: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return fights
    
    # Take in a fighter's URL and return a BeautifulSoup object of the fighter's page
    def get_fighter_page_url(self, fighter_url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a fighter's detail page.
        
        Args:
            fighter_url: URL to fighter's page
            
        Returns:
            BeautifulSoup object or None if failed
        """
        # Make a request to the fighter's URL (calls _make_request)
        response = self._make_request(fighter_url)
        # If the response is successful, return a BeautifulSoup object of the content
        if response:
            # Return a BeautifulSoup object of the content
            return BeautifulSoup(response.content, 'lxml')
        return None
    
    # Take in a fighter's URL and return a dictionary of the fighter's data
    def scrape_fighter(self, fighter_url: str) -> Optional[Dict]:
        """Scrape complete fighter data including stats and fight history.
        
        Args:
            fighter_url: URL to fighter's page
            
        Returns:
            Dictionary containing all fighter data or None if failed
        """
        # Print the fighter's URL
        logger.info(f"Scraping fighter: {fighter_url}")
        # Getting the BeautifulSoup object of the fighter's page
        soup = self.get_fighter_page_url(fighter_url)
        if not soup:
            return None
        
        # Creating a dictionary of the fighter's data
        fighter_data = {
            # Add the fighter's URL to the dictionary
            'url': fighter_url,
            # Add the fighter's statistics to the dictionary
            'stats': self._parse_fighter_stats(soup),
            # Add the fighter's fight history to the dictionary
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
        if event_url:
            return self._extract_fighters_from_event(event_url)
        else:
            # Return empty list - use get_fighters_from_events() instead
            return []
    
    # Caller can specify a limit on the number of events to retrieve
    def get_event_urls(self, limit: Optional[int] = None) -> List[str]:
        """Get list of event URLs from ufcstats.com events page.
        
        Args:
            limit: Optional limit on number of events to retrieve
            
        Returns:
            List of event URLs
        """
        # Array to hold the event URLs
        event_urls = []
        
        
        try:
            # Builds the URL for the events page
            events_page_url = f"{self.base_url}/statistics/events/completed?page=all"
            # Gets the BeautifulSoup object of the events page
            soup = self.get_fighter_page_url(events_page_url)
            # If the BeautifulSoup object is not found, log an error and return the empty array
            if not soup:
                logger.error("Failed to fetch events page")
                return event_urls
            
            # Find all event links - typically in a table with links to event details
            event_links = soup.find_all('a', href=lambda href: href and '/event-details/' in href)
            
            # Go through each event link and extract the event URL
            for link in event_links:
                # Convert the relative URL to an absolute URL
                event_url = urljoin(self.base_url, link.get('href'))
                # If the event URL is not in the array, add it to the array
                if event_url not in event_urls:
                    event_urls.append(event_url)
                # If the limit is specified and the number of event URLs is greater than or equal to the limit, break the loop
                if limit and len(event_urls) >= limit:
                    break
            # Logs the number of events found
            logger.info(f"Found {len(event_urls)} events")
        # If an error ocurrs log an error and return the empty array
        except Exception as e:
            logger.error(f"Error getting event URLs: {e}")
        
        # Returns the array of event URLs
        return event_urls
    
    # Returns a list of fighter URLs from a specific event page
    def _extract_fighters_from_event(self, event_url: str) -> List[str]:
        """Extract fighter URLs from a specific event page.
        
        Args:
            event_url: URL to event page
            
        Returns:
            List of fighter URLs found on the event page
        """
        # Array to hold the fighter URLs
        fighter_urls = []
        
        try:
            # Gets the BeautifulSoup object of the event page
            soup = self.get_fighter_page_url(event_url)
            # If the BeautifulSoup object is not found, log a warning and return the empty array
            if not soup:
                logger.warning(f"Failed to fetch event page: {event_url}")
                return fighter_urls
            
            # Finds all a tags with the href attribute that contains '/fighter-details/'
            fighter_links = soup.find_all('a', href=lambda href: href and '/fighter-details/' in href)
            # Go through each fighter link and extract the fighter URL
            for link in fighter_links:
                # Turns the relative URL into an absolute URL
                fighter_url = urljoin(self.base_url, link.get('href'))
                # If the fighter URL is not in the array, add it to the array
                if fighter_url not in fighter_urls:
                    fighter_urls.append(fighter_url)
            # Prints the number of fighter URLs found
            logger.info(f"Extracted {len(fighter_urls)} fighter URLs from event")
        # If an error ocurrs, logs an error
        except Exception as e:
            logger.error(f"Error extracting fighters from event {event_url}: {e}")
        
        return fighter_urls
    
    # User can specify a limit on the number of events to scrape, returns a list of unique fighter URLs
    def get_fighters_from_events(self, num_events: Optional[int] = None) -> List[str]:
        """Get fighter URLs from multiple events.
        
        Args:
            num_events: Number of recent events to scrape (None for all)
            
        Returns:
            List of unique fighter URLs
        """
        # Holds unique fighter URLs
        all_fighter_urls = set()
        
        # Logs the number of events to scrape
        logger.info(f"Getting fighter URLs from events (limit: {num_events})")
        
        # Get event URLs from the get_event_urls method
        event_urls = self.get_event_urls(limit=num_events)
        
        # Loops through each event URL and extracts the fighter URLs (starts at 1)
        for i, event_url in enumerate(event_urls, 1):
            # Logs the event number and URL
            logger.info(f"Processing event {i}/{len(event_urls)}: {event_url}")
            # Extracts the fighter URLs from the event URL
            fighter_urls = self._extract_fighters_from_event(event_url)
            # Adds the fighter_urls to the all_fighter_urls set
            all_fighter_urls.update(fighter_urls)
            
            # Rate limiting between events
            time.sleep(self.rate_limit_delay)
        
        # Converts the all_fighter_urls set to a list
        unique_urls = list(all_fighter_urls)
        # Logs the total number of unique fighter URLs found
        logger.info(f"Total unique fighter URLs found: {len(unique_urls)}")
        
        return unique_urls
    
    # Can specify a limit on the numer of events to scrape and where to save the master list
    def build_fighter_master_list(self, 
                                  num_events: Optional[int] = 50,
                                  save_path: Optional[str] = None) -> List[str]:
        """Build a master list of fighter URLs from events.
        
        Args:
            num_events: Number of recent events to scrape
            save_path: Optional path to save the master list (JSON file)
            
        Returns:
            List of unique fighter URLs
        """
        # Holds unique fighter URLs
        all_fighter_urls = set()
        # Logs a message saying the master fighter list is being built
        logger.info("Building master fighter list...")
        
        # Logs a message saying the fighter URLs are being collected from events
        logger.info("Collecting fighter URLs from events...")
        # Get fighters from events from the get_fighters_from_events method
        event_fighters = self.get_fighters_from_events(num_events=num_events)
        # Updates the all_fighter_urls set with the event_fighters
        all_fighter_urls.update(event_fighters)
        # Logs the number of unique fighters found from events
        logger.info(f"Found {len(event_fighters)} unique fighters from events")
        #Creats a list of all the unique fighter URLs
        master_list = list(all_fighter_urls)
        # Logs the number of unique fighter URLs in the master list
        logger.info(f"Master list contains {len(master_list)} unique fighter URLs")
        
        # Save master list if path provided
        if save_path:
            # Converts the save path to a Path object
            save_file = Path(save_path)
            # Ensures the save path exists
            save_file.parent.mkdir(parents=True, exist_ok=True)
            # Opens or creates the file
            with open(save_file, 'w') as f:
                # Writes the master list to the file in json format
                json.dump(master_list, f, indent=2)
            # Logs a message saying the master list was saved
            logger.info(f"Saved master list to {save_path}")
        
        return master_list
    
    def load_fighter_master_list(self, file_path: str) -> List[str]:
        """Load a previously saved master list of fighter URLs.
        
        Args:
            file_path: Path to JSON file containing fighter URLs
            
        Returns:
            List of fighter URLs
        """
        try:
            # Opens the file in read mode
            with open(file_path, 'r') as f:
                # Returns the file contents as a list of fighter URLs
                return json.load(f)
        # If the file is not found
        except FileNotFoundError:
            # Logs a message saying the master list file was not found
            logger.error(f"Master list file not found: {file_path}")
            # Returns an empty list
            return []
        # If an error occurs
        except Exception as e:
            # Logs an error and returns an empty list
            logger.error(f"Error loading master list: {e}")
            return []
    
    def save_fighter_data(self, fighter_data: Dict, format: str = 'json') -> None:
        """Save fighter data to file.
        
        Args:
            fighter_data: Dictionary containing fighter data
            format: Output format ('json' or 'csv')
        """
        # If the scraper failed or the stats are not found, log a warning and return
        if not fighter_data or 'stats' not in fighter_data:
            logger.warning("No valid fighter data to save")
            return
        # Gets the stats dictionary (otherwise it returns an empty dictionary) and gets the name of the fighter (otherwise it returns 'unknown')
        fighter_name = fighter_data.get('stats', {}).get('name', 'unknown')
        # Gets rid of any non-alphanumeric characters
        safe_name = "".join(c for c in fighter_name if c.isalnum() or c in (' ', '-', '_')).strip()
        # Replaces any spaces with underscores
        safe_name = safe_name.replace(' ', '_')
        
        # If the format is json, save the data to a json file
        if format == 'json':
            # Builds the filepath for the json file
            filename = self.raw_data_path / f"{safe_name}_data.json"
            # Open the file and write the data to it
            with open(filename, 'w') as f:
                # Converts the dictionary to a JSON string and writes it to the file
                json.dump(fighter_data, f, indent=2)
        # If the format is csv, save the data to a csv file
        elif format == 'csv':
            # Constructs the filepath for the stats csv file
            stats_filename = self.raw_data_path / f"{safe_name}_stats.csv"
            # If the stats are found, save them to a csv file
            if fighter_data.get('stats'):
                # Open the csv file in write mode and don't add a newline and give me a file object f
                with open(stats_filename, 'w', newline='') as f:
                    # Creates a dictionary writer object that writes the stats to the csv file and uses the keys of the stats dictionary
                    writer = csv.DictWriter(f, fieldnames=fighter_data['stats'].keys())
                    # Writes the headers to the csv file
                    writer.writeheader()
                    # Writes the stats to the csv file
                    writer.writerow(fighter_data['stats'])
            
            # Save fight history as separate CSV
            if fighter_data.get('fight_history'):
                # Constructs the filepath for the fight history csv file
                history_filename = self.raw_data_path / f"{safe_name}_fights.csv"
                # Open the csv file in write mode and don't add a newline and give me a file object f
                with open(history_filename, 'w', newline='') as f:
                    # Checks if the fight history is found
                    if fighter_data['fight_history']:
                        # Creates a dictionary writer object that writes the fight history to the csv file and uses the keys of the fight history dictionary
                        writer = csv.DictWriter(f, fieldnames=fighter_data['fight_history'][0].keys())
                        # Writes the headers to the csv file
                        writer.writeheader()
                        # Writes the fight history to the csv file
                        writer.writerows(fighter_data['fight_history'])
        # Logs a message saying the fighter data was saved
        logger.info(f"Saved fighter data for {fighter_name} to {filename if format == 'json' else stats_filename}")
    
    def _get_already_scraped_fighters(self) -> set:
        """Get set of fighter URLs that have already been scraped.
        
        Returns:
            Set of fighter URLs that already have data files
        """
        scraped = set()
        
        try:
            # Check for existing JSON files
            json_files = list(self.raw_data_path.glob("*_data.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if 'url' in data:
                            scraped.add(data['url'])
                except:
                    pass
        except Exception as e:
            logger.warning(f"Error checking already scraped fighters: {e}")
        
        return scraped
    
    # Takes in a list of fighter URLs and a save format and returns a list of dictionaries of the fighter's data
    def scrape_multiple_fighters(self, 
                                 fighter_urls: List[str], 
                                 save_format: str = 'json',
                                 skip_existing: bool = True,
                                 progress_file: Optional[str] = None) -> List[Dict]:
        """Scrape data for multiple fighters with progress tracking and resume capability.
        
        Args:
            fighter_urls: List of fighter URLs to scrape
            save_format: Format to save data ('json' or 'csv')
            skip_existing: If True, skip fighters that have already been scraped
            progress_file: Optional path to save progress (JSON file with list of completed URLs)
            
        Returns:
            List of fighter data dictionaries
        """
        # Collects all the fighter data
        all_fighter_data = []
        
        # Get already scraped fighters if skip_existing is True
        already_scraped = set()
        if skip_existing:
            already_scraped = self._get_already_scraped_fighters()
            logger.info(f"Found {len(already_scraped)} already scraped fighters. Skipping...")
        
        # Load progress file if it exists
        completed_urls = set()
        if progress_file and Path(progress_file).exists():
            try:
                with open(progress_file, 'r') as f:
                    completed_urls = set(json.load(f))
                logger.info(f"Loaded {len(completed_urls)} completed URLs from progress file")
            except Exception as e:
                logger.warning(f"Error loading progress file: {e}")
        
        # Filter out already scraped URLs
        urls_to_scrape = [url for url in fighter_urls 
                         if url not in already_scraped and url not in completed_urls]
        
        logger.info(f"Scraping {len(urls_to_scrape)} fighters (skipping {len(fighter_urls) - len(urls_to_scrape)} already done)")
        
        # Goes through each fighter URL and scrapes the data
        for i, url in enumerate(urls_to_scrape, 1):
            try:
                # Logs a message saying the fighter is being scraped
                logger.info(f"Scraping fighter {i}/{len(urls_to_scrape)}: {url}")
                # Scrapes the data for the fighter
                fighter_data = self.scrape_fighter(url)
                # If the fighter data is found, add it to the list of fighter data and save the data
                if fighter_data:
                    # Adds the fighter data to the list of fighter data
                    all_fighter_data.append(fighter_data)
                    # Saves the fighter data to a file
                    self.save_fighter_data(fighter_data, format=save_format)
                    # Mark as completed
                    completed_urls.add(url)
                else:
                    logger.warning(f"Failed to scrape fighter: {url}")
                
                # Save progress periodically (every 10 fighters or at the end)
                if progress_file and (i % 10 == 0 or i == len(urls_to_scrape)):
                    try:
                        with open(progress_file, 'w') as f:
                            json.dump(list(completed_urls), f, indent=2)
                    except Exception as e:
                        logger.warning(f"Error saving progress: {e}")
                
            except Exception as e:
                logger.error(f"Error scraping fighter {url}: {e}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        logger.info(f"Completed scraping. Successfully scraped {len(all_fighter_data)} fighters")
        
        return all_fighter_data


def main():
    """Example usage of the UFC scraper."""
   # Initializes the scraper
    scraper = UFCScraper()
    
    # Example: Scrape a specific fighter
    # fighter_url = "http://ufcstats.com/fighter-details/12345"
    # fighter_data = scraper.scrape_fighter(fighter_url)
    # if fighter_data:
    #     scraper.save_fighter_data(fighter_data)
    
    # Logs a message saying the scraper was initialized
    logger.info("UFC Scraper initialized. Use scraper.scrape_fighter(url) to scrape fighter data.")

# If the file is run directly, run the main function
if __name__ == "__main__":
    main()

