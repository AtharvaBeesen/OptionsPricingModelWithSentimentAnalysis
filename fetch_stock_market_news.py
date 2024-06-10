import requests
import json

def fetch_stock_market_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['articles']
    else:
        print("Failed to fetch news:", response.status_code)
        return None

def create_json_output(articles):
    output = []
    for article in articles:
        output.append({
            "title": article['title'],
            "date": article['publishedAt'],
            "source": article['source']['name'],
            "text": article['content']
        })
    return output

def main():
    api_key = "2afef13f6f4b4166aea1e34dd70d6083"
    query = "nvidia stock"
    articles = fetch_stock_market_news(api_key, query)
    if articles:
        output = create_json_output(articles)
        with open('nvidia_stock_news.json', 'w') as f:
            json.dump(output, f, indent=4)
        print("JSON file created successfully.")
    else:
        print("No articles fetched.")

if __name__ == "__main__":
    main()
