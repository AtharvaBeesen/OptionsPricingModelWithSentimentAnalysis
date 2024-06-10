![image](https://github.com/AtharvaBeesen/OptionsPricingModelWithSentimentAnalysis/assets/86427671/47f80562-c760-48fe-923f-0502a2ddb3b6)

# Options Pricing Model With Sentiment Analysis

This project involves fetching stock market news related to Nvidia, preprocessing the text data, analyzing sentiment using a fine-tuned language model, and pricing American options using the Longstaff-Schwartz method.

## Project Structure:

The project is structured into several main components:

1. **Fetching Stock Market News**: A Python script (`fetch_stock_market_news.py`) that retrieves stock market news related to Nvidia using the News API.

2. **Preprocessing Text Data**: A Python script (`preprocess_text_data.py`) that preprocesses the fetched news articles by removing HTML tags, special characters, punctuation, stopwords, and performs lemmatization.

3. **Sentiment Analysis**: Utilizes a fine-tuned GPT-2 language model to generate sentiment from the preprocessed news articles.

4. **American Options Pricing**: A Python class (`AmericanOptionsLSMC.py`) implementing the Longstaff-Schwartz method for pricing American options. It also includes functionality for sentiment-based adjustments to option parameters.

## Project Workflow:

1. **Fetching Stock Market News**:
   - Uses the News API to fetch news articles related to Nvidia stock.
   - Stores the fetched articles in a JSON file (`nvidia_stock_news.json`).

2. **Preprocessing Text Data**:
   - Reads the fetched articles from `nvidia_stock_news.json`.
   - Preprocesses each article by removing HTML tags, special characters, punctuation, stopwords, and performs lemmatization.
   - Saves the preprocessed articles in a new JSON file (`preprocessed_nvidia_stock_news.json`).

3. **Sentiment Analysis**:
   - Utilizes a fine-tuned GPT-2 language model to generate sentiment from the preprocessed news articles.
   - Performs simplistic sentiment analysis based on keywords such as 'positive' or 'negative'.

4. **American Options Pricing**:
   - Defines a class (`AmericanOptionsLSMC`) for pricing American options using the Longstaff-Schwartz method.
   - The class provides methods to calculate option price, delta, gamma, vega, rho, and theta.
   - It also includes functionality for sentiment-based adjustments to option parameters.

## Running the Project:

1. Ensure that Python and the required libraries (such as requests, nltk, transformers) are installed.
2. Execute the scripts in the following order:
   - `fetch_stock_market_news.py`
   - `preprocess_text_data.py`
   - `AmericanOptionsLSMC.py`
3. Follow the instructions in each script for any additional setup or configuration.
4. The main script (`AmericanOptionsLSMC.py`) demonstrates an example usage of the AmericanOptionsLSMC class with sentiment analysis.

## Dependencies:

- `requests`: For making HTTP requests to fetch news articles.
- `nltk`: For natural language processing tasks such as tokenization, stopword removal, and lemmatization.
- `transformers`: For loading and using the GPT-2 language model for sentiment analysis and text generation.

## Note:

- Ensure that the API keys required for accessing external services (such as the News API) are properly configured.
- This project provides a foundational structure for analyzing stock market news sentiment and pricing American options. Further enhancements and customization can be made

      S0 : float : initial stock/index level -> Potentially Adjusted due to sentiment analysis
      strike : float : strike price
      T : float : time to maturity (in year fractions)
      M : int : grid or granularity for time (in number of total points)
      r : float : constant risk-free short rate
      div : float : dividend yield
      sigma : float : volatility factor in diffusion term -> Potentially Adjusted due to sentiment analysis
