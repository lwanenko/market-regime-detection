import pandas as pd
import pywt
import numpy as np
import pandas_ta as ta
from genereted_features import *

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def ichimoku_cloud(df):
    high_prices = df['high']
    low_prices = df['low']
    df['tenkan_sen'] = (high_prices.rolling(window=9).max() + low_prices.rolling(window=9).min()) / 2
    df['kijun_sen'] = (high_prices.rolling(window=26).max() + low_prices.rolling(window=26).min()) / 2
    df['senkou_span_A'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_B'] = ((high_prices.rolling(window=52).max() + low_prices.rolling(window=52).min()) / 2).shift(26)
    return df


def fibonacci_retracement(df):
    max_price = df['high'].max()
    min_price = df['low'].min()
    difference = max_price - min_price
    df['fib_r1'] = max_price - 0.236 * difference  # 23.6%
    df['fib_r2'] = max_price - 0.382 * difference  # 38.2%
    df['fib_r3'] = max_price - 0.618 * difference  # 61.8%
    return df


def pivot_points(df):
    df['pivot_point'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['resistance1'] = 2 * df['pivot_point'] - df['low'].shift(1)
    df['support1'] = 2 * df['pivot_point'] - df['high'].shift(1)
    return df


def chaikin_volatility(df, ema_length=10, roc_length=10):
    high_low_diff = df['high'] - df['low']
    ema_high_low = high_low_diff.ewm(span=ema_length).mean()
    roc = ((ema_high_low - ema_high_low.shift(roc_length)) / ema_high_low.shift(roc_length)) * 100
    df['chaikin_volatility'] = roc
    return df


def volume_oscillator(df, short_span=12, long_span=26, signal_span=9):
    vo = ((df['volume'].rolling(window=short_span).mean() - df['volume'].rolling(window=long_span).mean()) / df['volume'].rolling(window=long_span).mean()) * 100
    vo_signal = vo.rolling(window=signal_span).mean()
    df['volume_oscillator'], df['vo_signal'] = vo, vo_signal
    return df


def keltner_channel(df, atr_length=20, multiplier=2):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(atr_length).mean()
    middle_line = df['close'].ewm(span=atr_length).mean()
    upper_line = middle_line + multiplier * atr
    lower_line = middle_line - multiplier * atr
    df['keltner_upper'], df['keltner_middle'], df['keltner_lower'] = upper_line, middle_line, lower_line
    return df


def wavelet_smooth(data, wavelet='db4', level=None):
    """ Enhanced Wavelet Smoothing with adjustable decomposition level and thresholding """

    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level]))/0.6745
    uthresh = sigma*np.sqrt(2*np.log(len(data)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    smoothed = pywt.waverec(coeffs, wavelet)
    smoothed = smoothed[:len(data)]
    return smoothed


def label_market_regimes(df, column, min_peak_distance=30, slope_std_multiplier=2):
    df['rolling_slope'] = df[column].diff().rolling(window=min_peak_distance, center=True).mean()
    rolling_slope_std = df['rolling_slope'].std()
    slope_threshold = rolling_slope_std * slope_std_multiplier
    df['market_mode'] = 'Flat'

    df.loc[df['rolling_slope'] > slope_threshold, 'market_mode'] = 'Bull'
    df.loc[df['rolling_slope'] < -slope_threshold, 'market_mode'] = 'Bear'
    df['market_mode'] = df['market_mode'].ffill().bfill()
    bull_bear_persist_filter = (df['market_mode'].shift() != df['market_mode']) & \
                               (df['market_mode'].shift(-1) != df['market_mode'])
    df.loc[bull_bear_persist_filter, 'market_mode'] = 'Flat'
    return df


def add_features(df, coef=1):
    #stupid features 
    df['median'] = df[["open", "high", "low", "close"]].median()
    df['co_prc_diff'] = (df['close'] - df['open'])/df['median']
    df['hl_prc_diff'] = (df['high'] - df['low'])/df['median']
    df['c2o'] =  df['close'] / df['open']
    df['h2l'] =  df['high'] / df['low']
    df['h2m'] =  df['high'] / df['median']
    df['l2m'] =  df['low'] / df['median']

    #base ta
    df['SMA'] = ta.sma(df['close'], length=14 * coef)  
    df['EMA'] = ta.ema(df['close'], length=14 * coef)  
    df['RSI_14'] = ta.rsi(df['close'], length=14 * coef)
    df['RSI_30'] = ta.rsi(df['close'], length=30 * coef)
    df['BB'] = ta.bbands(df['close'], length=20 * coef, std=2)['BBL_20_2.0']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])
    df['RSI'] = ta.rsi(df['close'], )
    df['STOCH'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHd_14_3_3']
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    df['PVO'] = ta.pvo(df['volume'])['PVO_12_26_9']
    ha_columns = ta.ha(df['open'], df['high'], df['low'], df['close'])
    df = pd.concat([df, ha_columns], axis=1)

    df = ichimoku_cloud(df)
    df = fibonacci_retracement(df)
    df = pivot_points(df)
    df = chaikin_volatility(df)
    df = volume_oscillator(df)
    df = keltner_channel(df)

    # generated
    df = renko(df, brick_size=int(round(df['ATR'].iloc[-1])))
    df = super_guppy(df)
    
    return df