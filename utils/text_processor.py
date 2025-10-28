"""
Text processing utilities for cleaning and preprocessing financial news text.
"""
import re
import string

def clean_text(text):
    """
    Clean text by removing URLs, special characters, and extra whitespace.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def remove_punctuation(text, keep_basic=True):
    """
    Remove or minimize punctuation from text.
    
    Args:
        text (str): Text to process
        keep_basic (bool): If True, keeps basic punctuation like periods and commas
        
    Returns:
        str: Text with punctuation removed/reduced
    """
    if keep_basic:
        # Remove only special punctuation, keep periods, commas
        remove_chars = string.punctuation.replace('.', '').replace(',', '')
        text = text.translate(str.maketrans('', '', remove_chars))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

def normalize_text(text):
    """
    Full text normalization pipeline.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    text = clean_text(text)
    text = text.lower()
    return text
