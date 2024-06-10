import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    return processed_text

def main():
    # Load Nvidia stock market articles from JSON file
    with open('nvidia_stock_news.json', 'r') as f:
        data = json.load(f)
    
    # Preprocess each article
    preprocessed_articles = []
    for article in data:
        processed_text = preprocess_text(article['text'])
        preprocessed_articles.append({
            "title": article['title'],
            "date": article['date'],
            "source": article['source'],
            "text": processed_text
        })

    # Save preprocessed articles to a new JSON file
    with open('preprocessed_nvidia_stock_news.json', 'w') as f:
        json.dump(preprocessed_articles, f, indent=4)

    print("Preprocessing complete. Preprocessed data saved to 'preprocessed_nvidia_stock_news.json'.")

if __name__ == "__main__":
    main()
