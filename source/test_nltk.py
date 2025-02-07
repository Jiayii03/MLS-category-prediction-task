import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

ABBR_DICT = {
   'ckg': 'checking', 'chk': 'check', 'dep': 'deposit', 'trns': 'transfer',
   'adv': 'advance', 'w/d': 'withdrawal', 'wd': 'withdrawal', 'xfer': 'transfer',
   'pmt': 'payment', 'txn': 'transaction', 'int': 'interest', 'intl': 'international',
   'intr': 'interest', 'fee': 'charge', 'chg': 'charge', 'pos': 'point of sale',
   'purch': 'purchase', 'atm': 'cash machine', 'atw': 'cash machine',
   'cd': 'certificate of deposit', 'cc': 'credit card', 'dc': 'debit card',
   'bal': 'balance', 'adj': 'adjustment', 'adjmt': 'adjustment', 'apmt': 'automatic payment',
   'auto': 'automatic', 'av': 'available', 'bk': 'bank', 'bkcard': 'bank card',
   'bkchg': 'bank charge', 'bkfee': 'bank fee', 'bkln': 'bank loan',
   'bkstmt': 'bank statement', 'bktrns': 'bank transfer', 'bkwd': 'bank withdrawal',
   'blnc': 'balance', 'bnk': 'bank', 'bnkchg': 'bank charge', 'n': "and", 'tx': 'transaction',
   'cb': 'chase bank', 'trsf': 'transfer', 'ref': 'reference', 'pymt': 'payment', 'pymnt': 'payment',
   'pmnt': 'payment', 'pw': '', 'ml': '', 'rcvd': 'received', 'dbt': 'debit', 'crd': 'card'
}

REMOVED_TERMS = {
   'ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia',
   'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms',
   'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa',
   'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy', 'rd',
   'date', 'card', 'st', 'ave', 'blvd', 'ln', 'dr', 'authorized', 'purchase'
}

KEPT_TERMS = {'7-eleven', 'walmart', 'target', 'costco', 'sams club'}

DATE_PATTERN = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b')
DIGITS_PATTERN = re.compile(r'\d+')
COLON_SLASH_PATTERN = re.compile(r'[:/]')
SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z\s-]')
REPEATED_SPACES = re.compile(r'\s+')

def extract_potential_entity(text):
   return DIGITS_PATTERN.sub('', text).strip()

def clean_normalize_text(text):
   text = text.lower()
   words = text.split()
   expanded_words = [ABBR_DICT.get(word.lower(), word) for word in words]
   text = ' '.join(expanded_words)
   
   text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
   
   named_entities = get_named_entities(text)
   
   cleaned_tokens = []
   for chunk in named_entities:
       if isinstance(chunk, nltk.Tree):
           if chunk.label() == 'ORGANIZATION':
               cleaned_tokens.append(' '.join([token for token, pos in chunk.leaves()]))
       else:
           word = chunk[0].lower()
           
         #   if word in KEPT_TERMS:
         #       cleaned_tokens.append(word)
         #       continue
               
           if (word not in stop_words and
               word not in REMOVED_TERMS and 
               not DATE_PATTERN.search(word) and
               not COLON_SLASH_PATTERN.search(word) and
               len(word) > 1 and
               not word.isdigit()):
               
               if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                   entity_name = extract_potential_entity(word)
                   if entity_name:
                       cleaned_tokens.append(entity_name)
               else:
                   cleaned_tokens.append(lemmatizer.lemmatize(word))
   
   result = ' '.join(cleaned_tokens)
   result = re.sub(r'\s+', ' ', result).strip()
   
   return result

def get_named_entities(text):
   tokens = nltk.word_tokenize(text)
   pos_tags = nltk.pos_tag(tokens)
   return nltk.ne_chunk(pos_tags)

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