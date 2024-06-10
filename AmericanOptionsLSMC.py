import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class AmericanOptionsLSMC:
    """Class for American options pricing using Longstaff-Schwartz (2001):
    "Valuing American Options by Simulation: A Simple Least-Squares Approach."
    """
    
    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, simulations):
        if option_type not in ['call', 'put']:
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if any(param < 0 for param in [S0, strike, T, r, div, sigma, simulations]):
            raise ValueError('Error: Negative inputs not allowed')
        
        self.option_type = option_type
        self.S0 = float(S0)
        self.strike = float(strike)
        self.T = float(T)
        self.M = int(M)
        self.r = float(r)
        self.div = float(div)
        self.sigma = float(sigma)
        self.simulations = int(simulations)
        
        self.time_unit = self.T / float(self.M)
        self.discount = np.exp((self.r - self.div) * self.time_unit)
        
        # Load the fine-tuned model and tokenizer for sentiment analysis
        self.tokenizer = GPT2Tokenizer.from_pretrained('trained_nvidia_model')
        self.model = GPT2LMHeadModel.from_pretrained('trained_nvidia_model')
        self.model.eval()

    def get_MCprice_matrix(self, seed=123):
        """Returns MC price matrix rows: time columns: price-path simulation"""
        np.random.seed(seed)
        MCprice_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        MCprice_matrix[0, :] = self.S0
        for t in range(1, self.M + 1):
            brownian = np.random.standard_normal(self.simulations // 2)
            brownian = np.concatenate((brownian, -brownian))
            MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :] *
                                    np.exp((self.r - self.div - self.sigma ** 2 / 2.) * self.time_unit +
                                           self.sigma * brownian * np.sqrt(self.time_unit)))
        return MCprice_matrix
    
    @property
    def MCprice_matrix(self):
        return self.get_MCprice_matrix()

    @property
    def MCpayoff(self):
        """Returns the inner-value of American Option"""
        if self.option_type == 'call':
            payoff = np.maximum(self.MCprice_matrix - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - self.MCprice_matrix, 0)
        return payoff

    @property
    def value_vector(self):
        value_matrix = np.zeros_like(self.MCpayoff)
        value_matrix[-1, :] = self.MCpayoff[-1, :]
        for t in range(self.M - 1, 0, -1):
            regression = np.polyfit(self.MCprice_matrix[t, :], value_matrix[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.MCprice_matrix[t, :])
            value_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation_value,
                                          self.MCpayoff[t, :],
                                          value_matrix[t + 1, :] * self.discount)
        return value_matrix[1, :] * self.discount

    @property
    def price(self):
        return np.sum(self.value_vector) / float(self.simulations)

    @property
    def delta(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0 + diff, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0 - diff, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def gamma(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0 + diff, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0 - diff, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
        return (myCall_1.delta - myCall_2.delta) / float(2. * diff)

    @property
    def vega(self):
        diff = self.sigma * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r, self.div, self.sigma + diff, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r, self.div, self.sigma - diff, self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def rho(self):
        diff = self.r * 0.01
        if (self.r - diff) < 0:
            myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r + diff, self.div, self.sigma, self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r, self.div, self.sigma, self.simulations)
            return (myCall_1.price - myCall_2.price) / float(diff)
        else:
            myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r + diff, self.div, self.sigma, self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T, self.M, self.r - diff, self.div, self.sigma, self.simulations)
            return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def theta(self):
        diff = 1 / 252.
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T + diff, self.M, self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, self.strike, self.T - diff, self.M, self.r, self.div, self.sigma, self.simulations)
        return (myCall_2.price - myCall_1.price) / float(2. * diff)

    def generate_sentiment(self, article):
        inputs = self.tokenizer.encode(article, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=512, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simplistic sentiment analysis based on keywords
        if 'positive' in generated_text:
            return 'positive'
        elif 'negative' in generated_text:
            return 'negative'
        else:
            return 'neutral'

    def adjust_parameters(self, sentiment):
        if sentiment == 'positive':
            self.S0 *= 1.05
            self.sigma *= 1.05
        elif sentiment == 'negative':
            self.S0 *= 0.95
            self.sigma *= 1.05

def main():
    # Example usage
    article = "Nvidia shares closed up 5% to $1,224.40 on Wednesday, giving the company a market cap above $3 trillion for the first time as investors continue to clamor for a piece of the company at the heart of the boom in generative artificial intelligence. Nvidia also passed Apple to become the second-largest public company behind Microsoft. Nvidia’s milestone is the latest stunning mark in a run that has seen the stock soar more than 3,224% over the past five years. The company will split its stock 10-for-1 later this month. Apple was the first U.S. company to reach a $3 trillion market cap during intraday trading in January 2022. Microsoft hit $3 trillion in market value in January 2024. Nvidia, which was founded in 1993, passed the $2 trillion valuation in February, and it only took roughly three months from there for it to pass $3 trillion. In May, Nvidia reported first-quarter earnings that showed demand for the company’s pricey and powerful graphics processing units, or GPUs, showed no sign of a slowdown. Nvidia reported overall sales of $26 billion, more than triple what it generated a year ago. Nvidia also beat Wall Street expectations for sales and earnings and said it would report revenue of about $28 billion in the current quarter. Nvidia’s surge in recent years has been powered by the tech industry’s need for its chips, which are used to develop and deploy big AI models such as the one at the heart of OpenAI’s ChatGPT. Companies such as Google, Microsoft, Meta, Amazon and OpenAI are buying billions of dollars worth of Nvidia’s GPUs." # input any news article.
    option_type = 'put'
    S0 = 36.
    strike = 40.
    T = 1.
    M = 50
    r = 0.06
    div = 0.06
    sigma = 0.2
    simulations = 10000

    american_option = AmericanOptionsLSMC(option_type, S0, strike, T, M, r, div, sigma, simulations)
    
    sentiment = american_option.generate_sentiment(article)
    print(f"Generated Sentiment: {sentiment}")

    american_option.adjust_parameters(sentiment)
    print(f"Adjusted S0: {american_option.S0}, Adjusted Sigma: {american_option.sigma}")
    print(f"Option Price: {american_option.price}")

if __name__ == "__main__":
    main()
