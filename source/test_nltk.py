import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, pos_tag

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Abbreviation and typo expansion dictionary
ABBR_DICT = {
    'ckg': 'checking',
    'chk': 'check',
    'dep': 'deposit',
    'trns': 'transfer',
    'adv': 'advance',
    'venmo': 'venmo transfer'
}

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_normalize_text(text):
    # Lowercase
    text = text.lower()
    
    # Expand abbreviations
    words = text.split()
    expanded_words = [ABBR_DICT.get(word, word) for word in words]
    text = ' '.join(expanded_words)
    
    # Tokenize and POS tag
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    # Perform Named Entity Recognition
    entities = ne_chunk(tagged)
    
    # Process tokens
    cleaned_tokens = []
    for subtree in entities:
        if type(subtree) == nltk.tree.Tree:
            # Remove personal names (PERSON) and locations (GPE)
            if subtree.label() in ['PERSON', 'GPE', 'DATE']:
                continue
            # Keep organization names
            elif subtree.label() == 'ORGANIZATION':
                cleaned_tokens.append(' '.join([leaf[0] for leaf in subtree.leaves()]))
        elif isinstance(subtree, tuple):
            token, pos = subtree
            # Remove punctuation, numbers, stopwords
            if (token not in string.punctuation 
                and not token.isdigit() 
                and token not in stop_words):
                # Lemmatize
                cleaned_tokens.append(lemmatizer.lemmatize(token))
    
    return ' '.join(cleaned_tokens)

# Test with a sample dummy transaction description
sample_text = "Deposit from John Doe to Chase Bank for loan repayment."
cleaned_text = clean_normalize_text(sample_text)

# Print results
print("Original Text: ", sample_text)
print("Anonymized Text: ", cleaned_text)