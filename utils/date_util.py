"""
Date Utility Tools to calculate Trading day-related functions
"""
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pandas as pd
import pytz
from datetime import date as datetime_dt


class DateUtil():
    """
    Utility class to get trading day calendar and time for a specific exchange.
    """

    def __init__(self, country):
        self.country = country
        self.exchange = ""
        self.timezone = ""

    def is_trading_day(self, date):
        """
        Check if the given date is a trading day for the exchange.
        """
        calendar = mcal.get_calendar(self.exchange)
        trading_days = calendar.valid_days(start_date=date, end_date=date)
        return not trading_days.empty

    def get_trading_day_diff(self, start_date, end_date):
        """
        Calculate the number of trading days between the given start and end dates.
        """
        calendar = mcal.get_calendar(self.exchange)
        trading_days = calendar.valid_days(start_date=start_date, end_date=end_date)
        return 0 if trading_days.empty else len(trading_days)

    def get_trading_time_diff(self, start_time, end_time):
        """
        :return units: minutes
        """
        start_time = self.str_check(start_time)
        end_time = self.str_check(end_time)
        start_time = self.transfer_current_timezone(start_time)
        end_time = self.transfer_current_timezone(end_time)
        if start_time.date() == end_time.date():
            return (end_time - start_time).seconds // 60
        days_diff = self.get_trading_day_diff(start_time.date(), end_time.date()) - 1
        return days_diff * (60 * 24) + (
                end_time.hour - start_time.hour - 1) * 60 + end_time.minute + start_time.minute

    def str_check(self, str_time):
        if isinstance(str_time, datetime) or isinstance(str_time, datetime_dt):
            return str_time
        if isinstance(str_time, str):
            try:
                if len(str_time) == 10:
                    return datetime.strptime(str_time, '%Y-%m-%d')
                if len(str_time) == 19:
                    return datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
            except:
                raise TypeError(f"Cannot recognize format: {str_time}")
        raise TypeError(f"Cannot recognize format: {str_time}")

    def get_cur_date(self):
        """
        Return the current date based on the timezone of the exchange.
        """
        return self.get_cur_date_time().date()

    def get_cur_date_time(self):
        """
        Return the current date and time based on the timezone of the exchange.
        """
        current_zone = pytz.timezone(self.timezone)
        time_location = datetime.now(current_zone)
        return time_location

    def get_former_trading_days(self, date, days):
        """
        Get the trading date that falls `days` before the given date.
        """
        calendar = mcal.get_calendar(self.exchange)
        last_days = calendar.schedule(start_date=date - timedelta(days=days + 5), end_date=date).index[-days - 1:]
        return last_days[0].date() if len(last_days) > 0 else date

    def get_latter_trading_days(self, date, days):
        """
        Get the trading date that falls `days` after the given date.
        """
        calendar = mcal.get_calendar(self.exchange)
        last_days = calendar.schedule(start_date=date, end_date=date + timedelta(days=days + 5)).index[days:]
        return last_days[0].date() if len(last_days) > 0 else date

    def transfer_current_timezone(self, current_date_time):
        """
        Convert the given datetime object to the timezone of the exchange.
        """
        return current_date_time.astimezone(pytz.timezone(self.timezone))


class DateUtilUSA(DateUtil):
    def __init__(self):
        super().__init__("USA")
        self.timezone = "US/Eastern"
        self.exchange = 'NYSE'

    def is_trading_time(self, current_date_time):
        """
        Check if the given time is within trading hours for the NYSE.
        """
        if current_date_time.hour == 9:
            return current_date_time.minute >= 30
        if current_date_time.hour == 16:
            return current_date_time.minute == 0
        return 10 <= current_date_time.hour < 16

    def is_open_market(self, current_time):
        """
        Check if the market is within the first 10 minutes after opening.
        """
        return (current_time.time() >= pd.Timestamp("09:30:00").time()) & (
                current_time.time() <= pd.Timestamp("09:40:00").time())

    def is_close_market(self, current_time):
        """
        Check if the market is within the last 10 minutes before closing.
        """
        return (current_time.time() >= pd.Timestamp("15:50:00").time()) & (
                current_time.time() <= pd.Timestamp("16:00:00").time())

    def is_open_close_market(self, current_time):
        """
        Check if the market is either within the first 10 minutes after opening
        or the last 10 minutes before closing.
        """
        return (self.is_open_market(current_time) | self.is_close_market(current_time))
