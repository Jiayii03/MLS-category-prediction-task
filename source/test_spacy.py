import spacy
import pandas as pd
import re
from tqdm import tqdm

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")

# Enable tqdm with Pandas
tqdm.pandas()

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

REMOVED_TERMS = {
    'ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 
    'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 
    'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 
    'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy', 'rd',
    'date', 'card'
}

KEPT_TERMS = {
    '7-eleven', '7eleven', 'walmart', 'circle k', 'target', 'costco', 'sams club'
}

# Regex patterns
DATE_PATTERN = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b')
DIGITS_PATTERN = re.compile(r'\d+')
COLON_SLASH_PATTERN = re.compile(r'[:/]')
SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z\s-]')
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
    # Convert to lowercase and expand abbreviations
    text = text.lower()
    
    # Check for kept terms before any processing
    for kept_term in KEPT_TERMS:
        if kept_term in text:
            return kept_term
        
    words = text.split()
    expanded_words = [ABBR_DICT.get(word.lower(), word) for word in words]
    text = ' '.join(expanded_words)

    # Remove special characters (except hyphens in company names)
    text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
    
    # Process with SpaCy
    doc = nlp(text)
    
    cleaned_tokens = []
    for token in doc:
        word = token.text.lower()
        print(token.text, token.ent_type_)
        
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

# Test cases
test_cases = [
    "POS PURCHASE POS19515501 1279102 VILLAGE MARKET Maryse Hemant MI",
    "PURCHASE AUTHORIZED ON 07/15 E3168ON SAM'S MART # CHARLOTTE NC S583197178931094 111",
    "PURCHASE AUTHORIZED ON 06/16 STARBUCKS STORE 50 CHARLOTTE NC S383167626089302 111",
    "Check Card Purchase / EMPOWER EMPOWER.ME CA Date 09/28/23 24011343271000026959174 6051 Card 4299",
    "PURCHASE AUTHORIZED ON 06/04 PAYPAL *POSHMARK 1036 CA S303156028350245 111",
    "PURCHASE AUTHORIZED ON 09/28 WAL-MART Wal-Mart Sup BETHLEHEM PA P000000787059413 111",
    "7-ELEVEN 35864 HENDERSON CO 1036",
    "7Eleven 35864 Henderson CO 1036",
]

for test in test_cases:
    print("\nOriginal:", test)
    print("Cleaned:", clean_normalize_text(test))
