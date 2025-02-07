from gensim.models import FastText

# Sample transaction descriptions
transactions = [
    "zelle transfer to john",
    "venmo payment received",
    "ach credit from chase",
    "7eleven purchase",
    "atm withdrawal"
]

# Tokenize text
tokenized_text = [text.split() for text in transactions]

# Train FastText on transaction descriptions
custom_fasttext = FastText(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Check word similarity in custom model
print("Words similar to 'Zelle':", custom_fasttext.wv.most_similar("zelle"))
print("Words similar to 'Venmo':", custom_fasttext.wv.most_similar("venmo"))
