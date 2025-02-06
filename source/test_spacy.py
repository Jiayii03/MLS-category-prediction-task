import spacy

# Load the large SpaCy model with NER capabilities
nlp = spacy.load("en_core_web_lg")

# Abbreviation expansion dictionary
ABBR_DICT = {
   'ckg': 'checking',
   'chk': 'check',
   'dep': 'deposit',
   'trns': 'transfer',
   'adv': 'advance',
    'w/d': 'withdrawal',
    'wd': 'withdrawal',
    'xfer': 'transfer',
    'pmt': 'payment',
    'txn': 'transaction',
    'int': 'interest',
    'intl': 'international',
    'intr': 'interest',
    'fee': 'charge',
    'chg': 'charge',
    'pos': 'purchase',
    'purch': 'purchase',
    'atm': 'cash machine',
    'atw': 'cash machine',
    'cd': 'certificate of deposit',
    'cc': 'credit card',
    'dc': 'debit card',
    'bal': 'balance',
    'adj': 'adjustment',
    'adjmt': 'adjustment',
    'apmt': 'automatic payment',
    'auto': 'automatic',
    'av': 'available',
    'bk': 'bank',
    'bkcard': 'bank card',
    'bkchg': 'bank charge',
    'bkfee': 'bank fee',
    'bkln': 'bank loan',
    'bkstmt': 'bank statement',
    'bktrns': 'bank transfer',
    'bkwd': 'bank withdrawal',
    'blnc': 'balance',
    'bnk': 'bank',
    'bnkchg': 'bank charge',
    'n': "and",
}

def clean_normalize_text(text):
   # Expand abbreviations
   words = text.split()
   expanded_words = [ABBR_DICT.get(word.lower(), word) for word in words]
   text = ' '.join(expanded_words)
   
   # Process with SpaCy
   doc = nlp(text.lower())
   
   # Process tokens
   cleaned_tokens = []
   for token in doc:
       # Skip entities: person names, locations, dates
       if token.ent_type_ in ['PERSON', 'GPE', 'DATE']:
           continue
       
       # Keep organization names
       if token.ent_type_ == 'ORG':
           cleaned_tokens.append(token.text)
           continue
       
       # Skip punctuation, stopwords, numbers
       if (not token.is_stop and 
           not token.is_punct and 
           not token.like_num):
           # Lemmatize
           cleaned_tokens.append(token.lemma_)
   
   return ' '.join(cleaned_tokens)

# Test with a sample dummy transaction description
sample_text = "Circle K dep from John Doe to Chase Bank for loan repayment."
anonymized_text = clean_normalize_text(sample_text)

# Print results
print("Original Text: ", sample_text)
print("Anonymized Text: ", anonymized_text)
