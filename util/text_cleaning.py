import spacy
import pandas as pd
import re
from tqdm import tqdm

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

# Enable tqdm with Pandas for progress bars
tqdm.pandas()

# Dictionary to expand common abbreviations in the text
ABBR_DICT = {
    'ckg': 'checking', 'chk': 'check', 'dep': 'deposit', 'trns': 'transfer',
    'adv': 'advance', 'w/d': 'withdrawal', 'wd': 'withdrawal', 'xfer': 'transfer',
    'pmt': 'payment', 'txn': 'transaction', 'int': 'interest', 'intl': 'international',
    'intr': 'interest', 'chg': 'charge', 'pos': 'point of sale',
    'purch': 'purchase', 'atm': 'cash machine', 'atw': 'cash machine',
    'cd': 'certificate of deposit', 'cc': 'credit card', 'dc': 'debit card',
    'bal': 'balance', 'adj': 'adjustment', 'adjmt': 'adjustment', 'apmt': 'automatic payment',
    'av': 'available', 'bk': 'bank', 'bkcard': 'bank card',
    'bkchg': 'bank charge', 'bkfee': 'bank fee', 'bkln': 'bank loan',
    'bkstmt': 'bank statement', 'bktrns': 'bank transfer', 'bkwd': 'bank withdrawal',
    'blnc': 'balance', 'bnk': 'bank', 'bnkchg': 'bank charge', 'n': "and", 'tx': 'transaction', 
    'cb': 'chase bank', 'trsf': 'transfer', 'ref': 'reference', 'pymt': 'payment', 'pymnt': 'payment', 
    'pmnt': 'payment', 'pw': '', 'ml': '', 'rcvd': 'received', 'dbt': 'debit', 'crd': 'card',
    'mar': 'mart', 'stor': 'store', 'sup': 'supermarket'
}

# Set of terms to remove from the text
REMOVED_TERMS = {
    'ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 
    'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 
    'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 
    'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy', 'rd',
    'date', 'card'
}

# Set of terms to keep in the text (e.g., specific company names)
KEPT_TERMS = {
    '7-eleven', '7eleven', '7 eleven', 'walmart', 'circle k', 'target', 'costco', 'sams club'
}

# Regex patterns for identifying dates, digits, colons/slashes, special characters, and repeated spaces
DATE_PATTERN = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b')
DIGITS_PATTERN = re.compile(r'\d+')
COLON_SLASH_PATTERN = re.compile(r'[:/]')
REPEATED_SPACES = re.compile(r'\s+')

def is_interleaved_alphanumeric(text):
    """Check if text has interleaved letters and numbers"""
    is_digit_prev = text[0].isdigit()
    transitions = 0
    for char in text[1:]:
        is_digit_curr = char.isdigit()
        if is_digit_curr != is_digit_prev:
            transitions += 1
        is_digit_prev = is_digit_curr
    return transitions > 2

def extract_potential_entity(text):
    """Extract letters from alphanumeric text if clearly separated"""
    if is_interleaved_alphanumeric(text):
        return None
    return DIGITS_PATTERN.sub('', text).strip()

def clean_normalize_text(text):
    """Clean and normalize text by expanding abbreviations, removing unwanted terms, and processing with SpaCy."""
    text = text.lower()
    
    # Check for kept terms before any processing
    for kept_term in KEPT_TERMS:
        if kept_term in text:
            return kept_term
        
    words = text.split()
    expanded_words = [ABBR_DICT.get(word.lower(), word) for word in words]
    text = ' '.join(expanded_words)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
    
    # Process text with SpaCy to tokenize and analyze entities
    doc = nlp(text)
    
    cleaned_tokens = []
    for token in doc:
        word = token.text.lower()
        
        # Remove entities: person names, locations, dates
        if token.ent_type_ in ['PERSON', 'GPE', 'DATE']:
            continue
        
        # Remove date patterns
        if DATE_PATTERN.search(word):
            continue

        # Remove words containing ":" or "/"
        if COLON_SLASH_PATTERN.search(word):
            continue
        
        # Skip if word is a state abbreviation
        if word in REMOVED_TERMS:
            continue
        
        # Keep organization names
        if token.ent_type_ == 'ORG':
            cleaned_tokens.append(token.text)
            continue
        
        # Handle alphanumeric words
        if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
            entity_name = extract_potential_entity(word)
            if entity_name:
                cleaned_tokens.append(entity_name.lower())
            continue
            
        # Skip punctuation, stopwords, numbers, and short words
        if (not token.is_punct and 
            not token.is_stop and 
            not token.like_num and 
            len(word) > 1):
            cleaned_tokens.append(token.lemma_)
            
    # Join tokens and clean up spaces
    result = ' '.join(cleaned_tokens)
    result = REPEATED_SPACES.sub(' ', result).strip()
    
    return result