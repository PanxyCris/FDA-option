from datetime import timedelta, datetime
import QuantLib as ql
import numpy as np
import yfinance as yf

from utils.date_util import DateUtilUSA

class Option():
    def __init__(self):
        pass

def calculate_american_option_price(option_type, underlying_price, strike_price, volatility, dividend_rate,
                                    maturity_date, risk_free_rate, calculation_date):
    """
    Calculate the price of an American option using QuantLib.

    Parameters:
    option_type (ql.Option.Call or ql.Option.Put): The type of the option (Call or Put).
    underlying_price (float): The current price of the underlying asset.
    strike_price (float): The strike price of the option.
    volatility (float): The volatility of the underlying asset.
    dividend_rate (float): The dividend yield of the underlying asset.
    maturity_date (ql.Date): The expiration date of the option.
    risk_free_rate (float): The risk-free interest rate.
    calculation_date (ql.Date): The current evaluation date.

    Returns:
    float: The theoretical price of the American option.
    """
    if not isinstance(maturity_date, ql.Date):
        maturity_date = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)

        # Convert calculation_date to QuantLib.Date if it's not already
    if not isinstance(calculation_date, ql.Date):
        calculation_date = ql.Date(calculation_date.day, calculation_date.month, calculation_date.year)

        # Set evaluation date
    ql.Settings.instance().evaluationDate = calculation_date

    # Define the option
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.AmericanExercise(calculation_date, maturity_date)
    american_option = ql.VanillaOption(payoff, exercise)

    # Construct the Black-Scholes process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, ql.Actual365Fixed()))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, ql.Actual365Fixed()))
    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, ql.NullCalendar(), volatility, ql.Actual365Fixed()))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)

    # Pricing engine
    steps = 200  # Number of steps for the binomial tree
    american_option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process, "crr", steps))

    # Calculate the price
    option_price = american_option.NPV()
    return option_price


def get_risk_free_rate(date_calculation, start_date=None, end_date=None, days=0):
    if start_date is None and end_date is None:
        days_diff = days
    else:
        if end_date is None:
            end_date = date_calculation.get_cur_date()
        days_diff = date_calculation.get_trading_day_diff(start_date, end_date)
    data = yf.download('^IRX', end=start_date + timedelta(days=1))
    return (1 + data['Close'].iloc[-1] / 100) ** (days_diff / 252) - 1


def get_third_friday(year, month):
    """Return the third Friday of the given month and year."""
    # The first day of the month
    first_day_of_month = datetime(year, month, 1)
    # First Friday of the month could be from day 1 to day 7
    # weekday() returns the day of the week as an integer (Monday is 0 and Sunday is 6)
    first_friday_of_month = first_day_of_month + timedelta(days=(4 - first_day_of_month.weekday() + 7) % 7)
    # Third Friday of the month is two weeks after the first Friday
    third_friday = first_friday_of_month + timedelta(weeks=2)
    return third_friday


def get_next_friday(given_date, can_equal_expiry):
    """Return the next Friday after the given date."""
    # The next Friday could be from 1 to 7 days away
    next_friday = given_date + timedelta(days=(4 - given_date.weekday() + 7) % 7)
    # If given date is Friday, add 7 days to get the next Friday
    if not can_equal_expiry and given_date.weekday() == 4:  # 4 represents Friday
        next_friday += timedelta(days=7)
    return next_friday


def get_maturity_date(given_date, option_type, can_equal_expiry):
    """
    Given a datetime, returns the expiration date of the option.

    If type == 'monthly', return 3rd Friday in that month. If 3rd Friday is equal or earlier than that datetime,
    choose next month's 3rd Friday.

    If type == 'weekly', return next Friday. If that date is Friday, return next Friday, not this Friday.
    """
    if option_type == 'monthly':
        third_friday = get_third_friday(given_date.year, given_date.month)
        if (third_friday <= given_date and not can_equal_expiry) or (third_friday < given_date and can_equal_expiry):
            # If the third Friday is earlier or equal, get the third Friday of the next month
            year = given_date.year + (given_date.month // 12)
            month = given_date.month % 12 + 1
            third_friday = get_third_friday(year, month)
        return third_friday
    elif option_type == 'weekly':
        return get_next_friday(given_date, can_equal_expiry)
    else:
        raise ValueError("Invalid option type. Please choose 'weekly' or 'monthly'.")


def get_historical_volatility(tickerObject, date, interval='1d'):
    hist = tickerObject.history(interval=interval, start=date - timedelta(days=365), end=date + timedelta(days=1))
    # Calculate daily returns
    returns = hist['Close'].pct_change().dropna()
    # Calculate the annualized volatility
    volatility = returns.std() * np.sqrt(252)
    return volatility


def buy_option(option_type, ticker, price, strike_price, start_date, option_frequency, can_equal_expiry=False):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    tickerObject = yf.Ticker(ticker)
    maturity_date = get_maturity_date(start_date, option_frequency, can_equal_expiry=can_equal_expiry)
    risk_free_rate = get_risk_free_rate(DateUtilUSA(), start_date=start_date, end_date=maturity_date)
    dividend_rate = tickerObject.info['dividendYield'] if 'dividendYield' in tickerObject.info else 0
    volatility = get_historical_volatility(tickerObject, start_date)
    option = Option()
    option.ticker = ticker
    option.option_frequency = option_frequency
    option.buy_stock_price = price
    option.strike_price = strike_price
    option.expiry_date = maturity_date
    option.dividend_rate = dividend_rate
    option.option_type = option_type
    option.option_price = calculate_american_option_price(option_type, price, strike_price, volatility, dividend_rate,
                                                          maturity_date, risk_free_rate, start_date)
    return option


def sell_option(option, price, start_date):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if start_date <= option.expiry_date:
        risk_free_rate = get_risk_free_rate(DateUtilUSA(), start_date=start_date, end_date=option.expiry_date)
        tickerObject = yf.Ticker(option.ticker)
        volatility = get_historical_volatility(tickerObject, start_date)
        bread = calculate_american_option_price(option.option_type, price, option.strike_price, volatility,
                                                option.dividend_rate,
                                                option.expiry_date, risk_free_rate, start_date)
    else:
        if option.option_type == ql.Option.Call:
            if price < option.strike_price:
                bread = 0
            else:
                bread = price - option.strike_price
        else:
            if price > option.strike_price:
                bread = 0
            else:
                bread = option.strike_price - price
    return bread
