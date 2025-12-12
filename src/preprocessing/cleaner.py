"""
Data preprocessing module for cleaning and normalizing scraped UFC data.
Consolidates individual JSON files into structured CSV datasets.

This module handles data cleaning, normalization, and consolidation
of raw fighter data into structured formats ready for feature engineering.
"""

# Used to read and write JSON files
import json
# Used to create dataframes (table-like structures for ML models)
import pandas as pd
# For numerical operations
import numpy as np
# For handling file paths
from pathlib import Path
# Used for type hints 
from typing import Dict, List, Optional, Tuple
# For printing to the console
import logging
# Used to convert fight dates to standard format
from datetime import datetime
# Extracts numbers from strings
import re
# Used to read configuration files
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
# Creates a logger object
logger = logging.getLogger(__name__)

# Class for cleaning and consolidating UFC fighter data
class DataCleaner:
    """Cleans and consolidates UFC fighter data from JSON files."""
    # Constructor for the DataCleaner class with a default configuration path
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data cleaner with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Calls the _load_config method to load the configuration
        self.config = self._load_config(config_path)
        
        # Looks inside the configuration dict for the paths to the raw and processed data
        paths_config = self.config.get('paths', {})
        # Gets the raw data path from the configuration and stores it as a Path object
        self.raw_data_path = Path(paths_config.get('raw_data', 'data/raw'))
        # Gets the processed data path from the configuration and stores it as a Path object
        self.processed_data_path = Path(paths_config.get('processed_data', 'data/processed'))
        # Creates the processed data path if it doesn't exist and checks if the parent directories exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Looks for preprocessing settings in the configuration dictionary elso {}
        preprocess_config = self.config.get('preprocessing', {})
        # Looks for the missing value strategy in the configuration dictionary else 'fill_zero'
        self.missing_value_strategy = preprocess_config.get('missing_value_strategy', 'fill_zero')
        # Looks for the normalize numeric setting in the configuration dictionary else True
        self.normalize_numeric = preprocess_config.get('normalize_numeric', True)
        # Gets the date format from the configuration dictionary else '%Y-%m-%d'
        self.date_format = preprocess_config.get('date_format', '%Y-%m-%d')
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            # Opens the configuration file in read mode and gives a file object f
            with open(config_path, 'r') as f:
                # Converts the YAML file to a dictionary and returns it
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def load_fighter_json_files(self) -> List[Dict]:
        """Load all fighter JSON files from raw data directory.
        
        Returns:
            List of fighter data dictionaries
        """
        # Creates an empty list to store the fighter data
        fighter_data = []
        # Looks for all JSON files in the raw data directory that end with '_data.json' and returns a list of Path objects
        json_files = list(self.raw_data_path.glob("*_data.json"))
        
        # Prints the number of JSON files found
        logger.info(f"Found {len(json_files)} fighter JSON files to process")
        # Loops through each JSON file
        for json_file in json_files:
            try:
                # Opens the JSON file in read mode and gives a file object f
                with open(json_file, 'r') as f:
                    # Loads the JSON file into a dictionary and appends it to the fighter data list
                    data = json.load(f)
                    # Appends the data to the fighter data list
                    fighter_data.append(data)
            except Exception as e:
                logger.warning(f"Error loading {json_file.name}: {e}")
        
        logger.info(f"Successfully loaded {len(fighter_data)} fighter files")
        return fighter_data
    
    def _parse_height(self, height_str: Optional[str]) -> Optional[float]:
        """Parse height string (e.g., "5' 10\"") to inches.
        
        Args:
            height_str: Height string
            
        Returns:
            Height in inches or None
        """
        # If the height string is None or 'N/A', return None
        if not height_str or height_str == 'N/A':
            return None
        
        try:
            # Match pattern like "5' 10\""
            match = re.match(r"(\d+)'\s*(\d+)\"", height_str)  
            # If the match is found, convert the feet and inches to inches
            if match:
                # Gets the feet from the match (first group)
                feet = int(match.group(1))
                # Gets the inches from the match (second group)
                inches = int(match.group(2))
                return feet * 12 + inches
        # If an exception is raised, pass
        except Exception:
            pass
        
        return None
    
    def _parse_weight(self, weight_str: Optional[str]) -> Optional[float]:
        """Parse weight string (e.g., "170 lbs.") to pounds.
        
        Args:
            weight_str: Weight string
            
        Returns:
            Weight in pounds or None
        """
        # If the weight string is None or 'N/A', return None
        if not weight_str or weight_str == 'N/A':
            return None
        
        try:
            # Extract number from string like "170 lbs."
            match = re.search(r'(\d+\.?\d*)', weight_str)
            # If the match is found, convert the weight to pounds
            if match:
                # Returns the weight as a float
                return float(match.group(1))
        except Exception:
            pass
        
        return None
    
    def _parse_reach(self, reach_str: Optional[str]) -> Optional[float]:
        """Parse reach string (e.g., "70\"") to inches.
        
        Args:
            reach_str: Reach string
            
        Returns:
            Reach in inches or None
        """
        # If the reach string is None or 'N/A', return None
        if not reach_str or reach_str == 'N/A':
            return None
        
        try:
            # Extract number from string like "70\""
            match = re.search(r'(\d+\.?\d*)', reach_str)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        
        return None
    
    # Method to parse the date string to the standard format
    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse date string to standard format.
        
        Args:
            date_str: Date string from scraped data
            
        Returns:
            Date in standard format (YYYY-MM-DD) or None
        """
        if not date_str or date_str == 'N/A':
            return None
        
        # Common date formats to try
        date_formats = [
            "%b. %d, %Y",  # "Nov. 15, 2025"
            "%B %d, %Y",   # "November 15, 2025"
            "%b %d, %Y",   # "Nov 15, 2025"
            "%Y-%m-%d",    # "2025-11-15"
            "%m/%d/%Y",    # "11/15/2025"
        ]
         # Loops through each date format
        for fmt in date_formats:
            try:
                # Converts the date string to a datetime object using the current date format
                dt = datetime.strptime(date_str.strip(), fmt)
                # Converts the datetime object to the standard format
                return dt.strftime(self.date_format)
            except ValueError:
                continue
        
        logger.debug(f"Could not parse date: {date_str}")
        return None
    
    # Method to parse the date of birth string to the standard format
    def _parse_dob(self, dob_str: Optional[str]) -> Optional[str]:
        """Parse date of birth string to standard format.
        
        Args:
            dob_str: Date of birth string
            
        Returns:
            Date in standard format (YYYY-MM-DD) or None
        """
        # Calls the _parse_date method to parse the date of birth string
        return self._parse_date(dob_str)
    
    def _normalize_numeric(self, value, default=0.0):
        """Normalize numeric values, handling None/null.
        
        Args:
            value: Value to normalize
            default: Default value if None/null
            
        Returns:
            Normalized numeric value
        """
        if value is None or value == 'N/A' or value == '':
            return default
        # Tries to convert the value to a float
        try:
            return float(value)
        # If an exception is raised, return the default value
        except (ValueError, TypeError):
            return default
    
    def clean_fighter_stats(self, fighter_data: List[Dict]) -> pd.DataFrame:
        """Clean and normalize fighter statistics.
        
        Args:
            fighter_data: List of fighter data dictionaries
            
        Returns:
            DataFrame with cleaned fighter statistics
        """
        cleaned_stats = []
        
        for fighter in fighter_data:
            if 'stats' not in fighter:
                continue
            
            stats = fighter['stats'].copy()
            
            # Add fighter URL for reference
            stats['fighter_url'] = fighter.get('url', '')
            
            # Parse physical attributes
            stats['height_inches'] = self._parse_height(stats.get('height'))
            stats['weight_lbs'] = self._parse_weight(stats.get('weight'))
            stats['reach_inches'] = self._parse_reach(stats.get('reach'))
            
            # Parse date of birth
            stats['dob_parsed'] = self._parse_dob(stats.get('dob'))
            
            # Normalize numeric fields
            numeric_fields = [
                'wins', 'losses', 'draws',
                'strikes_landed_per_min', 'striking_accuracy',
                'strikes_absorbed_per_min', 'striking_defense',
                'takedown_average', 'takedown_accuracy', 'takedown_defense',
                'submission_average'
            ]
            
            for field in numeric_fields:
                stats[field] = self._normalize_numeric(stats.get(field))
            
            # Calculate total fights
            stats['total_fights'] = stats.get('wins', 0) + stats.get('losses', 0) + stats.get('draws', 0)
            
            # Calculate win percentage
            total_fights = stats['total_fights']
            if total_fights > 0:
                stats['win_percentage'] = stats.get('wins', 0) / total_fights
            else:
                stats['win_percentage'] = 0.0
            
            cleaned_stats.append(stats)
        
        df = pd.DataFrame(cleaned_stats)
        
        # Handle missing values according to strategy
        if self.missing_value_strategy == 'fill_zero':
            # Selects all columns that are numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif self.missing_value_strategy == 'drop':
            df = df.dropna()
        elif self.missing_value_strategy == 'forward_fill':
            # Use ffill() for forward fill (pandas 2.0+ compatible)
            df = df.ffill()
        
        logger.info(f"Cleaned {len(df)} fighter statistics")
        return df
    
    def clean_fight_history(self, fighter_data: List[Dict]) -> pd.DataFrame:
        """Clean and normalize fight history data.
        
        Args:
            fighter_data: List of fighter data dictionaries
            
        Returns:
            DataFrame with cleaned fight history
        """
        all_fights = []
        
        for fighter in fighter_data:
            if 'fight_history' not in fighter or not fighter['fight_history']:
                continue
            
            fighter_name = fighter.get('stats', {}).get('name', 'Unknown')
            fighter_url = fighter.get('url', '')
            
            for fight in fighter['fight_history']:
                fight_record = {
                    'fighter_name': fighter_name,
                    'fighter_url': fighter_url,
                    'opponent': fight.get('opponent', ''),
                    'result': fight.get('result', '').lower(),
                    'date': self._parse_date(fight.get('date')),
                    'method': fight.get('method', ''),
                    'round': fight.get('round')
                }
                
                # Only include if we have essential data
                if fight_record['opponent'] and fight_record['result']:
                    all_fights.append(fight_record)
        
        df = pd.DataFrame(all_fights)
        
        # Sort by date (most recent first)
        if 'date' in df.columns:
            df = df.sort_values('date', ascending=False, na_position='last')
        
        logger.info(f"Cleaned {len(df)} fight records")
        return df
    
    def consolidate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Consolidate all fighter data into cleaned DataFrames.
        
        Returns:
            Tuple of (fighter_stats_df, fight_history_df)
        """
        logger.info("Starting data consolidation...")
        
        # Load all JSON files
        fighter_data = self.load_fighter_json_files()
        
        if not fighter_data:
            logger.warning("No fighter data found to process")
            return pd.DataFrame(), pd.DataFrame()
        
        # Clean fighter statistics
        logger.info("Cleaning fighter statistics...")
        fighter_stats_df = self.clean_fighter_stats(fighter_data)
        
        # Clean fight history
        logger.info("Cleaning fight history...")
        fight_history_df = self.clean_fight_history(fighter_data)
        
        logger.info("Data consolidation complete")
        return fighter_stats_df, fight_history_df
    
    def save_processed_data(self, 
                           fighter_stats_df: pd.DataFrame,
                           fight_history_df: pd.DataFrame) -> None:
        """Save processed data to CSV files.
        
        Args:
            fighter_stats_df: DataFrame with fighter statistics
            fight_history_df: DataFrame with fight history
        """
        # Save fighter statistics
        if not fighter_stats_df.empty:
            stats_file = self.processed_data_path / "all_fighters_stats.csv"
            fighter_stats_df.to_csv(stats_file, index=False)
            logger.info(f"Saved fighter statistics to {stats_file}")
            logger.info(f"  - {len(fighter_stats_df)} fighters")
            logger.info(f"  - {len(fighter_stats_df.columns)} columns")
        
        # Save fight history
        if not fight_history_df.empty:
            fights_file = self.processed_data_path / "all_fights.csv"
            fight_history_df.to_csv(fights_file, index=False)
            logger.info(f"Saved fight history to {fights_file}")
            logger.info(f"  - {len(fight_history_df)} fights")
            logger.info(f"  - {len(fight_history_df.columns)} columns")
    
    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete preprocessing pipeline.
        
        Returns:
            Tuple of (fighter_stats_df, fight_history_df)
        """
        logger.info("=" * 60)
        logger.info("Starting Data Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Consolidate data
        fighter_stats_df, fight_history_df = self.consolidate_data()
        
        # Save processed data
        self.save_processed_data(fighter_stats_df, fight_history_df)
        
        logger.info("=" * 60)
        logger.info("Data Preprocessing Complete!")
        logger.info("=" * 60)
        
        return fighter_stats_df, fight_history_df


def main():
    """Example usage of the data cleaner."""
    cleaner = DataCleaner()
    fighter_stats, fight_history = cleaner.process_all()
    
    print(f"\nProcessed {len(fighter_stats)} fighters and {len(fight_history)} fights")
    print(f"\nFighter stats columns: {list(fighter_stats.columns)}")
    print(f"\nFight history columns: {list(fight_history.columns)}")


if __name__ == "__main__":
    main()

