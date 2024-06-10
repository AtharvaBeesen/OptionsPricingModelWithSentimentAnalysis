# OptionsPricingModelWithSentimentAnalysis
    S0 : float : initial stock/index level -> Potentially Adjusted due to sentiment analysis
    strike : float : strike price
    T : float : time to maturity (in year fractions)
    M : int : grid or granularity for time (in number of total points)
    r : float : constant risk-free short rate
    div : float : dividend yield
    sigma : float : volatility factor in diffusion term -> Potentially Adjusted due to sentiment analysis
