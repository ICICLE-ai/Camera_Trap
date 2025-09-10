"""
Utilities for datetime normalization in camera trap data processing.
"""

from datetime import datetime
import re
try:
    from .loader_utils import get_settings_loader
except ImportError:
    from loader_utils import get_settings_loader

# Load datetime configuration from settings
_settings = get_settings_loader()
DATETIME_FORMATS = _settings.get_datetime_formats()
TARGET_DATETIME_FORMAT = _settings.get_target_datetime_format()

# Fallback formats in case settings file is not available
FALLBACK_DATETIME_FORMATS = [
    "%Y:%m:%d %H:%M:%S",        # Original format: "2016:01:08 10:35:00"
    "%Y-%m-%d %H:%M:%S",        # New format: "2016-01-08 10:35:00"
    "%Y/%m/%d %H:%M:%S",        # Alternative format: "2016/01/08 10:35:00"
    "%Y-%m-%d %H:%M:%S%z",      # With timezone: "2017-08-15 11:48:11+00:00"
    "%Y-%m-%dT%H:%M:%S",        # ISO format: "2018-03-23T04:22:12"
    "%Y-%m-%dT%H:%M:%S%z",      # ISO format with timezone: "2018-03-23T04:22:12+00:00"
    "%m/%d/%Y %H:%M",           # Format: "5/16/2016 2:38"
    "%m/%d/%Y %H:%M:%S",        # Format: "5/16/2016 2:38:45"
    "%d/%m/%Y %H:%M",           # European format: "19/09/2015 10:12"
    "%d/%m/%Y %H:%M:%S",        # European format: "19/09/2015 10:12:30"
    "%Y-%m-%d",                 # Date only: "2016-01-08"
    "%Y:%m:%d",                 # Date only with colons: "2016:01:08"
    "%Y/%m/%d",                 # Date only with slashes: "2016/01/08"
]

# Use fallback if settings not loaded
if not DATETIME_FORMATS:
    DATETIME_FORMATS = FALLBACK_DATETIME_FORMATS
    print("⚠️  Warning: Using fallback datetime formats - settings.yaml not found")

if not TARGET_DATETIME_FORMAT:
    TARGET_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


def normalize_datetime(datetime_str):
    """
    Normalize datetime string to the standard format: "YYYY:MM:DD HH:MM:SS"
    
    Args:
        datetime_str (str): Input datetime string in various formats
        
    Returns:
        str: Normalized datetime string in "YYYY:MM:DD HH:MM:SS" format
        
    Raises:
        ValueError: If the datetime string cannot be parsed
    """
    if not datetime_str or not isinstance(datetime_str, str):
        raise ValueError(f"Invalid datetime input: {datetime_str}")
    
    # Clean the input string
    datetime_str = datetime_str.strip()
    
    # Handle timezone suffixes by removing them for parsing
    # Remove timezone information that might cause issues
    datetime_str_clean = re.sub(r'([+-]\d{2}:?\d{2}|Z)$', '', datetime_str)
    
    # Try to parse the datetime with each format
    for fmt in DATETIME_FORMATS:
        try:
            # Parse the datetime
            dt = datetime.strptime(datetime_str_clean, fmt)
            
            # If the original format didn't include time, default to 00:00:00
            if '%H' not in fmt:
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Format to target format
            return dt.strftime(TARGET_DATETIME_FORMAT)
            
        except ValueError:
            continue
    
    # If all formats failed, try to handle some edge cases
    try:
        # Try to handle ISO format with 'T' separator
        if 'T' in datetime_str:
            iso_clean = datetime_str.replace('T', ' ')
            iso_clean = re.sub(r'([+-]\d{2}:?\d{2}|Z)$', '', iso_clean)
            for fmt in DATETIME_FORMATS:
                if 'T' not in fmt:
                    try:
                        dt = datetime.strptime(iso_clean, fmt)
                        return dt.strftime(TARGET_DATETIME_FORMAT)
                    except ValueError:
                        continue
        
        # Try to handle fractional seconds by truncating
        if '.' in datetime_str:
            parts = datetime_str.split('.')
            if len(parts) == 2:
                base = parts[0]
                # Try without fractional seconds
                for fmt in DATETIME_FORMATS:
                    if '.%f' not in fmt and '%z' not in fmt:
                        try:
                            dt = datetime.strptime(base, fmt)
                            return dt.strftime(TARGET_DATETIME_FORMAT)
                        except ValueError:
                            continue
                            
    except Exception:
        pass
    
    # If all parsing attempts failed, raise an error
    raise ValueError(f"Could not parse datetime string: '{datetime_str}'. "
                     f"Supported formats: {DATETIME_FORMATS}")


def validate_datetime_format(datetime_str):
    """
    Validate if a datetime string is in the target format.
    
    Args:
        datetime_str (str): Datetime string to validate
        
    Returns:
        bool: True if the datetime is in the target format, False otherwise
    """
    try:
        datetime.strptime(datetime_str, TARGET_DATETIME_FORMAT)
        return True
    except (ValueError, TypeError):
        return False


def batch_normalize_datetimes(datetime_list):
    """
    Normalize a list of datetime strings.
    
    Args:
        datetime_list (list): List of datetime strings
        
    Returns:
        list: List of normalized datetime strings
        
    Raises:
        ValueError: If any datetime string cannot be parsed
    """
    normalized = []
    errors = []
    
    for i, dt_str in enumerate(datetime_list):
        try:
            normalized.append(normalize_datetime(dt_str))
        except ValueError as e:
            errors.append(f"Index {i}: {e}")
    
    if errors:
        raise ValueError(f"Failed to normalize {len(errors)} datetime strings:\n" + "\n".join(errors))
    
    return normalized


# Test function for development/debugging
def test_datetime_normalization():
    """Test the datetime normalization function with various formats."""
    test_cases = [
        "2016:01:08 10:35:00",
        "2016-01-08 10:35:00",
        "2016/01/08 10:35:00",
        "2016-01-08",
        "2016:01:08",
        "2016/01/08",
        "2001-02-28 00:00:00.000",
        "2020-01-08 14:44:06+00:00",
        "2016-01-08T10:35:00",
        "2016-01-08T10:35:00.123456",
        "2016-01-08T10:35:00+00:00",
        "2016-01-08T10:35:00Z",
        "01/08/2016 10:35:00",
        "08/01/2016 10:35:00",
        "01/08/2016",
        "08/01/2016",
    ]
    
    print("Testing datetime normalization:")
    for test_case in test_cases:
        try:
            result = normalize_datetime(test_case)
            print(f"✓ '{test_case}' -> '{result}'")
        except ValueError as e:
            print(f"✗ '{test_case}' -> ERROR: {e}")


if __name__ == "__main__":
    test_datetime_normalization()
