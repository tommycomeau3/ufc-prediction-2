"""
Feature engineering module for creating ML features from processed UFC data.
Creates features comparing fighters for each fight, including basic stats,
advanced metrics, recent form, and head-to-head comparisons.
"""
# Pandas for data manipulation
import pandas as pd
# NumPy for numerical operations
import numpy as np
# Pathlib for file path manipulation
from pathlib import Path
# difflib for fuzzy name matching
import difflib
# Type hints for the function arguments and return values
from typing import Dict, List, Optional, Tuple
# Logging for error and warning messages
import logging
# Datetime for date and time operations
from datetime import datetime
# YAML for configuration file
import yaml
# Scikit-learn for machine learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Warnings for ignoring warnings
import warnings
# Ignores warnings
warnings.filterwarnings('ignore')

# Sets up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates ML features from processed UFC fighter data."""
    
    # Initializes the FeatureEngineer class
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Loads the configuration from the YAML file and makes it available as self.config
        self.config = self._load_config(config_path)
        
        # Get paths from self.config otherwise its become {}
        paths_config = self.config.get('paths', {})
       # Creates a Path object for the processed data path
        self.processed_data_path = Path(paths_config.get('processed_data', 'data/processed'))
        # Creates a Path object for the features data path
        self.features_data_path = Path(paths_config.get('features_data', 'data/features'))
        # Creates the features data path if it doesn't exist
        self.features_data_path.mkdir(parents=True, exist_ok=True)
        
        # Get feature engineering settings
        features_config = self.config.get('features', {})
        self.include_basic_stats = features_config.get('include_basic_stats', True)
        self.include_record = features_config.get('include_record', True)
        self.include_finish_types = features_config.get('include_finish_types', True)
        self.include_striking_stats = features_config.get('include_striking_stats', True)
        self.include_takedown_stats = features_config.get('include_takedown_stats', True)
        self.include_recent_form = features_config.get('include_recent_form', True)
        self.recent_form_window = features_config.get('recent_form_window', 5)
        self.include_physical_stats = features_config.get('include_physical_stats', True)
        self.include_advantage_metrics = features_config.get('include_advantage_metrics', True)
        self.include_streaks = features_config.get('include_streaks', True)
        self.include_fight_frequency = features_config.get('include_fight_frequency', True)
        self.include_avg_fight_duration = features_config.get('include_avg_fight_duration', True)
        
        # Data storage
        self.fighter_stats_df = None
        self.fight_history_df = None
        self.feature_matrix_df = None
        self.scaler = None
        
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

    def _get_similar_fighter_names(self, query: str, n: int = 5) -> List[str]:
        """Return similar fighter names from the database for 'Did you mean?' suggestions.

        Args:
            query: The name that was not found
            n: Maximum number of suggestions to return

        Returns:
            List of similar names from fighter_stats_df
        """
        if self.fighter_stats_df is None:
            return []
        choices = self.fighter_stats_df['name'].astype(str).tolist()
        return difflib.get_close_matches(query.strip(), choices, n=n, cutoff=0.4)

    def _resolve_fighter_name(self, name: str) -> str:
        """Resolve a fighter name to the canonical database name (handles case/whitespace).

        Returns:
            Canonical name if found
        Raises:
            ValueError: If not found, with suggestions in the message
        """
        if self.fighter_stats_df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")

        stripped = name.strip()
        if not stripped:
            raise ValueError("Fighter name cannot be empty.")

        # Exact match
        exact = self.fighter_stats_df[self.fighter_stats_df['name'] == stripped]
        if len(exact) > 0:
            return exact.iloc[0]['name']

        # Case-insensitive match
        lower = stripped.lower()
        mask = (
            self.fighter_stats_df['name'].astype(str).str.strip().str.lower() == lower
        )
        matches = self.fighter_stats_df[mask]
        if len(matches) > 0:
            return matches.iloc[0]['name']

        # Not found - get suggestions
        suggestions = self._get_similar_fighter_names(stripped, n=5)
        if suggestions:
            suggestions_str = ", ".join(suggestions)
            raise ValueError(
                f"Fighter '{stripped}' not found. Did you mean: {suggestions_str}?"
            )
        raise ValueError(
            f"Fighter '{stripped}' not found in database. "
            "Check spelling and try again, or the fighter may not be in our dataset."
        )

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed fighter stats and fight history.
        
        Returns:
            Tuple of (fighter_stats_df, fight_history_df)
        """
        stats_file = self.processed_data_path / "all_fighters_stats.csv"
        fights_file = self.processed_data_path / "all_fights.csv"
        
        if not stats_file.exists():
            raise FileNotFoundError(f"Fighter stats file not found: {stats_file}")
        if not fights_file.exists():
            raise FileNotFoundError(f"Fight history file not found: {fights_file}")
        
        logger.info("Loading processed data...")
        fighter_stats_df = pd.read_csv(stats_file)
        fight_history_df = pd.read_csv(fights_file)
        
        # Convert date columns
        if 'dob_parsed' in fighter_stats_df.columns:
            fighter_stats_df['dob_parsed'] = pd.to_datetime(fighter_stats_df['dob_parsed'], errors='coerce')
        if 'date' in fight_history_df.columns:
            fight_history_df['date'] = pd.to_datetime(fight_history_df['date'], errors='coerce')
        
        self.fighter_stats_df = fighter_stats_df
        self.fight_history_df = fight_history_df
        
        logger.info(f"Loaded {len(fighter_stats_df)} fighters and {len(fight_history_df)} fights")
        
        return fighter_stats_df, fight_history_df
    
    def _calculate_age_at_fight(self, dob: pd.Timestamp, fight_date: pd.Timestamp) -> Optional[float]:
        """Calculate fighter age at time of fight.
        
        Args:
            dob: Date of birth
            fight_date: Fight date
            
        Returns:
            Age in years or None
        """
        if pd.isna(dob) or pd.isna(fight_date):
            return None
        try:
            age_delta = fight_date - dob
            return age_delta.days / 365.25
        except:
            return None
    
    def _calculate_recent_form(self, fighter_name: str, fight_date: pd.Timestamp, 
                               window: int = 5) -> Dict:
        """Calculate recent form metrics for a fighter.
        
        Args:
            fighter_name: Fighter name
            fight_date: Date of current fight
            window: Number of recent fights to consider
            
        Returns:
            Dictionary with recent form metrics
        """
        # Get fights before this date
        past_fights = self.fight_history_df[
            (self.fight_history_df['fighter_name'] == fighter_name) &
            (self.fight_history_df['date'] < fight_date) &
            (self.fight_history_df['result'].isin(['win', 'loss']))
        ].sort_values('date', ascending=False).head(window)
        
        if len(past_fights) == 0:
            return {
                'recent_wins': 0,
                'recent_losses': 0,
                'recent_win_rate': 0.0,
                'recent_fight_count': 0
            }
        
        recent_wins = (past_fights['result'] == 'win').sum()
        recent_losses = (past_fights['result'] == 'loss').sum()
        recent_win_rate = recent_wins / len(past_fights) if len(past_fights) > 0 else 0.0
        
        return {
            'recent_wins': recent_wins,
            'recent_losses': recent_losses,
            'recent_win_rate': recent_win_rate,
            'recent_fight_count': len(past_fights)
        }
    
    def _calculate_streak(self, fighter_name: str, fight_date: pd.Timestamp) -> Dict:
        """Calculate current win/loss streak for a fighter.
        
        Args:
            fighter_name: Fighter name
            fight_date: Date of current fight
            
        Returns:
            Dictionary with streak information
        """
        # Get most recent fights before this date
        past_fights = self.fight_history_df[
            (self.fight_history_df['fighter_name'] == fighter_name) &
            (self.fight_history_df['date'] < fight_date) &
            (self.fight_history_df['result'].isin(['win', 'loss']))
        ].sort_values('date', ascending=False)
        
        if len(past_fights) == 0:
            return {'win_streak': 0, 'loss_streak': 0}
        
        # Count consecutive wins/losses from most recent
        win_streak = 0
        loss_streak = 0
        
        for _, fight in past_fights.iterrows():
            if fight['result'] == 'win':
                if loss_streak > 0:
                    break
                win_streak += 1
            elif fight['result'] == 'loss':
                if win_streak > 0:
                    break
                loss_streak += 1
        
        return {'win_streak': win_streak, 'loss_streak': loss_streak}
    
    def _calculate_fight_frequency(self, fighter_name: str, fight_date: pd.Timestamp) -> Optional[float]:
        """Calculate average days between fights.
        
        Args:
            fighter_name: Fighter name
            fight_date: Date of current fight
            
        Returns:
            Average days between fights or None
        """
        past_fights = self.fight_history_df[
            (self.fight_history_df['fighter_name'] == fighter_name) &
            (self.fight_history_df['date'] < fight_date)
        ].sort_values('date', ascending=False)
        
        if len(past_fights) < 2:
            return None
        
        # Calculate average days between fights
        date_diffs = []
        for i in range(len(past_fights) - 1):
            diff = (past_fights.iloc[i]['date'] - past_fights.iloc[i + 1]['date']).days
            if diff > 0:
                date_diffs.append(diff)
        
        if len(date_diffs) == 0:
            return None
        
        return np.mean(date_diffs)
    
    def _extract_finish_types(self, fighter_name: str, fight_date: pd.Timestamp) -> Dict:
        """Extract finish type counts from fight history.
        
        Args:
            fighter_name: Fighter name
            fight_date: Date of current fight
            
        Returns:
            Dictionary with finish type counts
        """
        past_fights = self.fight_history_df[
            (self.fight_history_df['fighter_name'] == fighter_name) &
            (self.fight_history_df['date'] < fight_date) &
            (self.fight_history_df['result'] == 'win')
        ]
        
        ko_tko_count = past_fights['method'].str.contains('KO/TKO', case=False, na=False).sum()
        submission_count = past_fights['method'].str.contains('Submission', case=False, na=False).sum()
        decision_count = past_fights['method'].str.contains('Decision', case=False, na=False).sum()
        
        return {
            'ko_tko_wins': ko_tko_count,
            'submission_wins': submission_count,
            'decision_wins': decision_count
        }
    
    def create_fight_features(self) -> pd.DataFrame:
        """Create feature matrix for each fight with fighter comparisons.
        
        Returns:
            DataFrame with features for each fight
        """
        if self.fighter_stats_df is None or self.fight_history_df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        logger.info("Creating fight features...")
        
        # Filter to completed fights (exclude 'next' fights)
        completed_fights = self.fight_history_df[
            self.fight_history_df['result'].isin(['win', 'loss'])
        ].copy()
        
        logger.info(f"Processing {len(completed_fights)} completed fights")
        
        fight_features = []
        
        for idx, fight in completed_fights.iterrows():
            fighter1_name = fight['fighter_name']
            fighter2_name = fight['opponent']
            fight_date = fight['date']
            result = fight['result']  # 'win' means fighter1 won
            
            # Get fighter stats
            fighter1_stats = self.fighter_stats_df[
                self.fighter_stats_df['name'] == fighter1_name
            ].iloc[0] if len(self.fighter_stats_df[self.fighter_stats_df['name'] == fighter1_name]) > 0 else None
            
            fighter2_stats = self.fighter_stats_df[
                self.fighter_stats_df['name'] == fighter2_name
            ].iloc[0] if len(self.fighter_stats_df[self.fighter_stats_df['name'] == fighter2_name]) > 0 else None
            
            if fighter1_stats is None or fighter2_stats is None:
                continue
            
            # Initialize feature dictionary
            features = {
                'fight_date': fight_date,
                'fighter1_name': fighter1_name,
                'fighter2_name': fighter2_name,
                'target': 1 if result == 'win' else 0  # 1 = fighter1 wins
            }
            
            # Basic stats
            if self.include_basic_stats:
                features['f1_total_fights'] = fighter1_stats.get('total_fights', 0)
                features['f2_total_fights'] = fighter2_stats.get('total_fights', 0)
                features['f1_win_percentage'] = fighter1_stats.get('win_percentage', 0)
                features['f2_win_percentage'] = fighter2_stats.get('win_percentage', 0)
                features['f1_wins'] = fighter1_stats.get('wins', 0)
                features['f2_wins'] = fighter2_stats.get('wins', 0)
                features['f1_losses'] = fighter1_stats.get('losses', 0)
                features['f2_losses'] = fighter2_stats.get('losses', 0)
            
            # Physical attributes
            if self.include_physical_stats:
                features['f1_height'] = fighter1_stats.get('height_inches', 0)
                features['f2_height'] = fighter2_stats.get('height_inches', 0)
                features['height_diff'] = features.get('f1_height', 0) - features.get('f2_height', 0)
                
                features['f1_reach'] = fighter1_stats.get('reach_inches', 0)
                features['f2_reach'] = fighter2_stats.get('reach_inches', 0)
                features['reach_diff'] = features.get('f1_reach', 0) - features.get('f2_reach', 0)
                
                features['f1_weight'] = fighter1_stats.get('weight_lbs', 0)
                features['f2_weight'] = fighter2_stats.get('weight_lbs', 0)
                features['weight_diff'] = features.get('f1_weight', 0) - features.get('f2_weight', 0)
                
                # Age at fight
                f1_age = self._calculate_age_at_fight(fighter1_stats.get('dob_parsed'), fight_date)
                f2_age = self._calculate_age_at_fight(fighter2_stats.get('dob_parsed'), fight_date)
                features['f1_age'] = f1_age if f1_age else 0
                features['f2_age'] = f2_age if f2_age else 0
                features['age_diff'] = features.get('f1_age', 0) - features.get('f2_age', 0)
            
            # Striking stats
            if self.include_striking_stats:
                features['f1_strikes_landed_per_min'] = fighter1_stats.get('strikes_landed_per_min', 0)
                features['f2_strikes_landed_per_min'] = fighter2_stats.get('strikes_landed_per_min', 0)
                features['f1_strikes_absorbed_per_min'] = fighter1_stats.get('strikes_absorbed_per_min', 0)
                features['f2_strikes_absorbed_per_min'] = fighter2_stats.get('strikes_absorbed_per_min', 0)
                features['f1_striking_accuracy'] = fighter1_stats.get('striking_accuracy', 0)
                features['f2_striking_accuracy'] = fighter2_stats.get('striking_accuracy', 0)
                features['f1_striking_defense'] = fighter1_stats.get('striking_defense', 0)
                features['f2_striking_defense'] = fighter2_stats.get('striking_defense', 0)
            
            # Takedown stats
            if self.include_takedown_stats:
                features['f1_takedown_accuracy'] = fighter1_stats.get('takedown_accuracy', 0)
                features['f2_takedown_accuracy'] = fighter2_stats.get('takedown_accuracy', 0)
                features['f1_takedown_defense'] = fighter1_stats.get('takedown_defense', 0)
                features['f2_takedown_defense'] = fighter2_stats.get('takedown_defense', 0)
                features['f1_takedown_average'] = fighter1_stats.get('takedown_average', 0)
                features['f2_takedown_average'] = fighter2_stats.get('takedown_average', 0)
            
            # Recent form
            if self.include_recent_form:
                f1_form = self._calculate_recent_form(fighter1_name, fight_date, self.recent_form_window)
                f2_form = self._calculate_recent_form(fighter2_name, fight_date, self.recent_form_window)
                
                features['f1_recent_win_rate'] = f1_form['recent_win_rate']
                features['f2_recent_win_rate'] = f2_form['recent_win_rate']
                features['f1_recent_wins'] = f1_form['recent_wins']
                features['f2_recent_wins'] = f2_form['recent_wins']
            
            # Streaks
            if self.include_streaks:
                f1_streak = self._calculate_streak(fighter1_name, fight_date)
                f2_streak = self._calculate_streak(fighter2_name, fight_date)
                
                features['f1_win_streak'] = f1_streak['win_streak']
                features['f2_win_streak'] = f2_streak['win_streak']
                features['f1_loss_streak'] = f1_streak['loss_streak']
                features['f2_loss_streak'] = f2_streak['loss_streak']
            
            # Finish types
            if self.include_finish_types:
                f1_finishes = self._extract_finish_types(fighter1_name, fight_date)
                f2_finishes = self._extract_finish_types(fighter2_name, fight_date)
                
                features['f1_ko_tko_wins'] = f1_finishes['ko_tko_wins']
                features['f2_ko_tko_wins'] = f2_finishes['ko_tko_wins']
                features['f1_submission_wins'] = f1_finishes['submission_wins']
                features['f2_submission_wins'] = f2_finishes['submission_wins']
            
            # Fight frequency
            if self.include_fight_frequency:
                features['f1_avg_days_between_fights'] = self._calculate_fight_frequency(fighter1_name, fight_date) or 0
                features['f2_avg_days_between_fights'] = self._calculate_fight_frequency(fighter2_name, fight_date) or 0
            
            # Days since last fight
            f1_days = self._calculate_days_since_last_fight(fighter1_name, fight_date)
            f2_days = self._calculate_days_since_last_fight(fighter2_name, fight_date)
            features['f1_days_since_last_fight'] = f1_days if f1_days is not None else 0
            features['f2_days_since_last_fight'] = f2_days if f2_days is not None else 0
            
            # Layoff penalty
            features['f1_layoff_penalty'] = self._calculate_layoff_penalty(f1_days)
            features['f2_layoff_penalty'] = self._calculate_layoff_penalty(f2_days)
            
            # Strength of schedule
            f1_sos = self._calculate_strength_of_schedule(fighter1_name, fight_date)
            f2_sos = self._calculate_strength_of_schedule(fighter2_name, fight_date)
            features['f1_strength_of_schedule'] = f1_sos if f1_sos is not None else 0
            features['f2_strength_of_schedule'] = f2_sos if f2_sos is not None else 0
            
            # Quality adjusted win percentage
            features['f1_quality_adjusted_win_pct'] = features.get('f1_win_percentage', 0) * (f1_sos if f1_sos is not None else 0.5)
            features['f2_quality_adjusted_win_pct'] = features.get('f2_win_percentage', 0) * (f2_sos if f2_sos is not None else 0.5)
            
            # Advantage metrics (fighter1 - fighter2)
            if self.include_advantage_metrics:
                features['win_pct_advantage'] = features.get('f1_win_percentage', 0) - features.get('f2_win_percentage', 0)
                features['striking_advantage'] = features.get('f1_strikes_landed_per_min', 0) - features.get('f2_strikes_landed_per_min', 0)
                features['striking_differential'] = (features.get('f1_strikes_landed_per_min', 0) - features.get('f1_strikes_absorbed_per_min', 0)) - \
                                                    (features.get('f2_strikes_landed_per_min', 0) - features.get('f2_strikes_absorbed_per_min', 0))
                features['reach_advantage'] = features.get('f1_reach', 0) - features.get('f2_reach', 0)
                features['strength_of_schedule_advantage'] = features.get('f1_strength_of_schedule', 0) - features.get('f2_strength_of_schedule', 0)
                features['quality_adjusted_win_pct_advantage'] = features.get('f1_quality_adjusted_win_pct', 0) - features.get('f2_quality_adjusted_win_pct', 0)
            
            fight_features.append(features)
        
        self.feature_matrix_df = pd.DataFrame(fight_features)
        logger.info(f"Created feature matrix with {len(self.feature_matrix_df)} fights and {len(self.feature_matrix_df.columns)} features")
        
        return self.feature_matrix_df
    
    def create_single_fight_features(self, fighter1_name: str, fighter2_name: str, 
                                     fight_date: str) -> pd.DataFrame:
        """Create features for a single future fight between two fighters.
        
        Args:
            fighter1_name: Name of first fighter
            fighter2_name: Name of second fighter
            fight_date: Date of the fight (YYYY-MM-DD format)
            
        Returns:
            DataFrame with features for the single fight
        """
        # Load data if not already loaded
        if self.fighter_stats_df is None or self.fight_history_df is None:
            self.load_processed_data()
        
        # Parse fight date
        if isinstance(fight_date, str):
            fight_date = pd.to_datetime(fight_date)
        
        # Resolve fighter names (handles case/whitespace, suggests alternatives if not found)
        fighter1_name = self._resolve_fighter_name(fighter1_name)
        fighter2_name = self._resolve_fighter_name(fighter2_name)

        fighter1_stats = self.fighter_stats_df[
            self.fighter_stats_df['name'] == fighter1_name
        ].iloc[0]
        fighter2_stats = self.fighter_stats_df[
            self.fighter_stats_df['name'] == fighter2_name
        ].iloc[0]
        
        # Initialize feature dictionary
        features = {
            'fight_date': fight_date,
            'fighter1_name': fighter1_name,
            'fighter2_name': fighter2_name,
            'target': 0  # No target for future fights
        }
        
        # Basic stats
        if self.include_basic_stats:
            features['f1_total_fights'] = fighter1_stats.get('total_fights', 0)
            features['f2_total_fights'] = fighter2_stats.get('total_fights', 0)
            features['f1_win_percentage'] = fighter1_stats.get('win_percentage', 0)
            features['f2_win_percentage'] = fighter2_stats.get('win_percentage', 0)
            features['f1_wins'] = fighter1_stats.get('wins', 0)
            features['f2_wins'] = fighter2_stats.get('wins', 0)
            features['f1_losses'] = fighter1_stats.get('losses', 0)
            features['f2_losses'] = fighter2_stats.get('losses', 0)
        
        # Physical attributes
        if self.include_physical_stats:
            features['f1_height'] = fighter1_stats.get('height_inches', 0)
            features['f2_height'] = fighter2_stats.get('height_inches', 0)
            features['height_diff'] = features.get('f1_height', 0) - features.get('f2_height', 0)
            
            features['f1_reach'] = fighter1_stats.get('reach_inches', 0)
            features['f2_reach'] = fighter2_stats.get('reach_inches', 0)
            features['reach_diff'] = features.get('f1_reach', 0) - features.get('f2_reach', 0)
            
            features['f1_weight'] = fighter1_stats.get('weight_lbs', 0)
            features['f2_weight'] = fighter2_stats.get('weight_lbs', 0)
            features['weight_diff'] = features.get('f1_weight', 0) - features.get('f2_weight', 0)
            
            # Age at fight
            f1_age = self._calculate_age_at_fight(fighter1_stats.get('dob_parsed'), fight_date)
            f2_age = self._calculate_age_at_fight(fighter2_stats.get('dob_parsed'), fight_date)
            features['f1_age'] = f1_age if f1_age else 0
            features['f2_age'] = f2_age if f2_age else 0
            features['age_diff'] = features.get('f1_age', 0) - features.get('f2_age', 0)
        
        # Striking stats
        if self.include_striking_stats:
            features['f1_strikes_landed_per_min'] = fighter1_stats.get('strikes_landed_per_min', 0)
            features['f2_strikes_landed_per_min'] = fighter2_stats.get('strikes_landed_per_min', 0)
            features['f1_strikes_absorbed_per_min'] = fighter1_stats.get('strikes_absorbed_per_min', 0)
            features['f2_strikes_absorbed_per_min'] = fighter2_stats.get('strikes_absorbed_per_min', 0)
            features['f1_striking_accuracy'] = fighter1_stats.get('striking_accuracy', 0)
            features['f2_striking_accuracy'] = fighter2_stats.get('striking_accuracy', 0)
            features['f1_striking_defense'] = fighter1_stats.get('striking_defense', 0)
            features['f2_striking_defense'] = fighter2_stats.get('striking_defense', 0)
        
        # Takedown stats
        if self.include_takedown_stats:
            features['f1_takedown_accuracy'] = fighter1_stats.get('takedown_accuracy', 0)
            features['f2_takedown_accuracy'] = fighter2_stats.get('takedown_accuracy', 0)
            features['f1_takedown_defense'] = fighter1_stats.get('takedown_defense', 0)
            features['f2_takedown_defense'] = fighter2_stats.get('takedown_defense', 0)
            features['f1_takedown_average'] = fighter1_stats.get('takedown_average', 0)
            features['f2_takedown_average'] = fighter2_stats.get('takedown_average', 0)
        
        # Recent form
        if self.include_recent_form:
            f1_form = self._calculate_recent_form(fighter1_name, fight_date, self.recent_form_window)
            f2_form = self._calculate_recent_form(fighter2_name, fight_date, self.recent_form_window)
            
            features['f1_recent_win_rate'] = f1_form['recent_win_rate']
            features['f2_recent_win_rate'] = f2_form['recent_win_rate']
            features['f1_recent_wins'] = f1_form['recent_wins']
            features['f2_recent_wins'] = f2_form['recent_wins']
        
        # Streaks
        if self.include_streaks:
            f1_streak = self._calculate_streak(fighter1_name, fight_date)
            f2_streak = self._calculate_streak(fighter2_name, fight_date)
            
            features['f1_win_streak'] = f1_streak['win_streak']
            features['f2_win_streak'] = f2_streak['win_streak']
            features['f1_loss_streak'] = f1_streak['loss_streak']
            features['f2_loss_streak'] = f2_streak['loss_streak']
        
        # Finish types
        if self.include_finish_types:
            f1_finishes = self._extract_finish_types(fighter1_name, fight_date)
            f2_finishes = self._extract_finish_types(fighter2_name, fight_date)
            
            features['f1_ko_tko_wins'] = f1_finishes['ko_tko_wins']
            features['f2_ko_tko_wins'] = f2_finishes['ko_tko_wins']
            features['f1_submission_wins'] = f1_finishes['submission_wins']
            features['f2_submission_wins'] = f2_finishes['submission_wins']
        
        # Fight frequency
        if self.include_fight_frequency:
            features['f1_avg_days_between_fights'] = self._calculate_fight_frequency(fighter1_name, fight_date) or 0
            features['f2_avg_days_between_fights'] = self._calculate_fight_frequency(fighter2_name, fight_date) or 0
        
        # Days since last fight
        f1_days = self._calculate_days_since_last_fight(fighter1_name, fight_date)
        f2_days = self._calculate_days_since_last_fight(fighter2_name, fight_date)
        features['f1_days_since_last_fight'] = f1_days if f1_days is not None else 0
        features['f2_days_since_last_fight'] = f2_days if f2_days is not None else 0
        
        # Layoff penalty
        features['f1_layoff_penalty'] = self._calculate_layoff_penalty(f1_days)
        features['f2_layoff_penalty'] = self._calculate_layoff_penalty(f2_days)
        
        # Strength of schedule
        f1_sos = self._calculate_strength_of_schedule(fighter1_name, fight_date)
        f2_sos = self._calculate_strength_of_schedule(fighter2_name, fight_date)
        features['f1_strength_of_schedule'] = f1_sos if f1_sos is not None else 0
        features['f2_strength_of_schedule'] = f2_sos if f2_sos is not None else 0
        
        # Quality adjusted win percentage
        features['f1_quality_adjusted_win_pct'] = features.get('f1_win_percentage', 0) * (f1_sos if f1_sos is not None else 0.5)
        features['f2_quality_adjusted_win_pct'] = features.get('f2_win_percentage', 0) * (f2_sos if f2_sos is not None else 0.5)
        
        # Advantage metrics (fighter1 - fighter2)
        if self.include_advantage_metrics:
            features['win_pct_advantage'] = features.get('f1_win_percentage', 0) - features.get('f2_win_percentage', 0)
            features['striking_advantage'] = features.get('f1_strikes_landed_per_min', 0) - features.get('f2_strikes_landed_per_min', 0)
            features['striking_differential'] = (features.get('f1_strikes_landed_per_min', 0) - features.get('f1_strikes_absorbed_per_min', 0)) - \
                                                (features.get('f2_strikes_landed_per_min', 0) - features.get('f2_strikes_absorbed_per_min', 0))
            features['reach_advantage'] = features.get('f1_reach', 0) - features.get('f2_reach', 0)
            features['strength_of_schedule_advantage'] = features.get('f1_strength_of_schedule', 0) - features.get('f2_strength_of_schedule', 0)
            features['quality_adjusted_win_pct_advantage'] = features.get('f1_quality_adjusted_win_pct', 0) - features.get('f2_quality_adjusted_win_pct', 0)
        
        # Create DataFrame with single row
        features_df = pd.DataFrame([features])
        
        return features_df
    
    def _calculate_days_since_last_fight(self, fighter_name: str, fight_date: pd.Timestamp) -> Optional[int]:
        """Calculate days since last fight.
        
        Args:
            fighter_name: Fighter name
            fight_date: Date of current fight
            
        Returns:
            Days since last fight or None
        """
        past_fights = self.fight_history_df[
            (self.fight_history_df['fighter_name'] == fighter_name) &
            (self.fight_history_df['date'] < fight_date)
        ].sort_values('date', ascending=False)
        
        if len(past_fights) == 0:
            return None
        
        last_fight_date = past_fights.iloc[0]['date']
        days_diff = (fight_date - last_fight_date).days
        return days_diff if days_diff >= 0 else None
    
    def _calculate_layoff_penalty(self, days_since_last_fight: Optional[int]) -> float:
        """Calculate layoff penalty based on days since last fight.
        Optimal range is 120-180 days (4-6 months), with optimal at 150 days (5 months).
        
        Args:
            days_since_last_fight: Days since last fight, or None
            
        Returns:
            Layoff penalty score (lower is better, optimal is 0)
        """
        if days_since_last_fight is None:
            return 1.0  # Penalty for no fight history
        
        # Optimal range: 120-180 days (4-6 months)
        # Optimal point: 150 days (5 months)
        optimal_min = 120
        optimal_max = 180
        optimal_point = 150
        
        if days_since_last_fight < optimal_min:
            # Too short: penalty increases as days decrease
            penalty = 1.0 - (days_since_last_fight / optimal_min)
        elif days_since_last_fight > optimal_max:
            # Too long: penalty increases as days increase
            excess = days_since_last_fight - optimal_max
            # Penalty grows slower for longer layoffs
            penalty = excess / 365.0  # Normalize by year
            penalty = min(penalty, 2.0)  # Cap at 2.0
        else:
            # Optimal range: penalty based on distance from optimal point
            distance_from_optimal = abs(days_since_last_fight - optimal_point)
            max_distance = max(optimal_point - optimal_min, optimal_max - optimal_point)
            penalty = (distance_from_optimal / max_distance) * 0.5  # Max 0.5 penalty in optimal range
        
        return max(0.0, penalty)  # Ensure non-negative
    
    def _calculate_strength_of_schedule(self, fighter_name: str, fight_date: pd.Timestamp) -> Optional[float]:
        """Calculate strength of schedule (average opponent win percentage).
        
        Args:
            fighter_name: Fighter name
            fight_date: Date of current fight
            
        Returns:
            Average opponent win percentage or None
        """
        past_fights = self.fight_history_df[
            (self.fight_history_df['fighter_name'] == fighter_name) &
            (self.fight_history_df['date'] < fight_date)
        ]
        
        if len(past_fights) == 0:
            return None
        
        opponent_win_pcts = []
        for _, fight in past_fights.iterrows():
            opponent_name = fight['opponent']
            opponent_stats = self.fighter_stats_df[
                self.fighter_stats_df['name'] == opponent_name
            ]
            
            if len(opponent_stats) > 0:
                win_pct = opponent_stats.iloc[0].get('win_percentage', 0)
                if win_pct is not None and not np.isnan(win_pct):
                    opponent_win_pcts.append(win_pct)
        
        if len(opponent_win_pcts) == 0:
            return None
        
        return np.mean(opponent_win_pcts)
    
    def scale_features(self, features_df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale/standardize features for ML models.
        
        Args:
            features_df: DataFrame with features
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {method} method...")
        
        # Identify numeric columns to scale (exclude metadata columns)
        exclude_cols = ['fight_date', 'fighter1_name', 'fighter2_name', 'target']
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(scale_cols) == 0:
            logger.warning("No numeric columns to scale")
            return features_df
        
        # Create scaled dataframe
        scaled_df = features_df.copy()
        
        # Fit scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaled_df[scale_cols] = self.scaler.fit_transform(features_df[scale_cols])
        
        logger.info(f"Scaled {len(scale_cols)} features")
        
        return scaled_df
    
    def save_scaler(self, filename: str = "feature_scaler.pkl") -> None:
        """Save the fitted scaler to disk.
        
        Args:
            filename: Name of the scaler file
        """
        import joblib
        if self.scaler is None:
            raise ValueError("No scaler to save. Call scale_features() first.")
        
        scaler_file = self.features_data_path / filename
        joblib.dump(self.scaler, scaler_file)
        logger.info(f"Saved scaler to {scaler_file}")
    
    def load_scaler(self, filename: str = "feature_scaler.pkl") -> None:
        """Load a previously saved scaler from disk.
        
        Args:
            filename: Name of the scaler file
        """
        import joblib
        scaler_file = self.features_data_path / filename
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
        
        self.scaler = joblib.load(scaler_file)
        logger.info(f"Loaded scaler from {scaler_file}")
    
    def transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using a previously fitted scaler.
        
        Args:
            features_df: DataFrame with features to transform
            
        Returns:
            DataFrame with transformed features
        """
        if self.scaler is None:
            raise ValueError("No scaler loaded. Call load_scaler() first.")
        
        # Identify numeric columns to scale (exclude metadata columns)
        exclude_cols = ['fight_date', 'fighter1_name', 'fighter2_name', 'target']
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(scale_cols) == 0:
            logger.warning("No numeric columns to transform")
            return features_df
        
        # Create transformed dataframe
        transformed_df = features_df.copy()
        transformed_df[scale_cols] = self.scaler.transform(features_df[scale_cols])
        
        logger.info(f"Transformed {len(scale_cols)} features")
        
        return transformed_df
    
    def save_features(self, features_df: pd.DataFrame, filename: str = "fight_features.csv") -> None:
        """Save feature matrix to CSV file.
        
        Args:
            features_df: DataFrame with features
            filename: Output filename
        """
        output_file = self.features_data_path / filename
        features_df.to_csv(output_file, index=False)
        logger.info(f"Saved features to {output_file}")
    
    def engineer_features(self, scale: bool = True, scale_method: str = 'standard') -> pd.DataFrame:
        """Run complete feature engineering pipeline.
        
        Args:
            scale: Whether to scale features
            scale_method: Scaling method ('standard' or 'minmax')
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("=" * 60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 60)
        
        # Load processed data
        self.load_processed_data()
        
        # Create fight features
        features_df = self.create_fight_features()
        
        # Scale features if requested
        if scale:
            features_df = self.scale_features(features_df, method=scale_method)
            # Save the scaler for future use
            self.save_scaler()
        
        # Save features
        self.save_features(features_df)
        
        logger.info("=" * 60)
        logger.info("Feature Engineering Complete!")
        logger.info("=" * 60)
        logger.info(f"Created {len(features_df)} fight records with {len(features_df.columns)} features")
        
        return features_df


def main():
    """Example usage of the feature engineer."""
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(scale=True, scale_method='standard')
    
    print(f"\nFeature matrix shape: {features_df.shape}")
    print(f"\nFeature columns: {list(features_df.columns)}")
    print(f"\nSample features:\n{features_df.head()}")


if __name__ == "__main__":
    main()

