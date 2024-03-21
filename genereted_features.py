import pandas as pd
import pandas_ta as ta
import numpy as np


#%% renko
class Renko:
    # https://github.com/mr-easy/streaming_indicators/blob/main/src/streaming_indicators/streaming_indicators.py
    def __init__(self, start_price=None):
        self.bricks = []
        self.current_direction = 0
        self.brick_end_price = start_price
        self.pwick = 0   # positive wick
        self.nwick = 0   # negative wick
        self.brick_num = 0
        self.value = None
        
    def _create_brick(self, direction, brick_size, price):
        self.brick_end_price = round(self.brick_end_price + direction*brick_size,2)
        brick = {
            'direction': direction,
            'brick_num': self.brick_num,
            'wick_size': self.nwick if direction==1 else self.pwick,
            'brick_size': brick_size,
            'brick_end_price': self.brick_end_price,
            'price': price
        }
        self.bricks.append(brick)
        self.brick_num += 1
        self.current_direction = direction
        self.pwick = 0
        self.nwick = 0
        return brick        
        
    def update(self, price, brick_size):
        if(self.brick_end_price is None):
            self.brick_end_price = price
            return None
        if(brick_size is None): return None
        bricks = None
        change = round(price - self.brick_end_price, 2)
        self.pwick = max(change, self.pwick)
        self.nwick = min(-change, self.nwick)
        if(self.current_direction == 0):
            direction = 0
            if(change >= brick_size): direction = 1
            elif(-change >= brick_size): direction = -1
            if(direction != 0):
                #print("firect brick direction:", str(direction))
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(direction, brick_size, price) for i in range(num_bricks)]
                
        elif(self.current_direction == 1):
            if(change >= brick_size):
                # more bricks in +1 direction
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(1, brick_size, price) for i in range(num_bricks)]
                
            elif(-change >= 2 * brick_size):
                # reverse direction to -1
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(-1, brick_size, price) for i in range(num_bricks-1)]
                
        elif(self.current_direction == -1):
            if(-change >= brick_size):
                # more bricks in -1 direction
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(-1, brick_size, price) for i in range(num_bricks)]
                
            elif(change >= 2 * brick_size):
                # reverse direction to +1
                num_bricks = int(abs(change)//brick_size)
                bricks = [self._create_brick(1, brick_size, price) for i in range(num_bricks-1)]

        self.value = bricks
        return bricks

def renko(df, brick_size):
    renko_chart = Renko()
    directions = []
    brick_numbers = []
    wick_sizes = []
    brick_end_prices = []

    for index, row in df.iterrows():
        price = row['close']
        new_bricks = renko_chart.update(price, brick_size)

        if new_bricks:
            last_brick = new_bricks[-1]
            directions.append(last_brick['direction'])
            brick_numbers.append(last_brick['brick_num'])
            wick_sizes.append(last_brick['wick_size'])
            brick_end_prices.append(last_brick['brick_end_price'])
        else:
            directions.append(None)
            brick_numbers.append(None)
            wick_sizes.append(None)
            brick_end_prices.append(None)

    df['renko_direction'] = directions
    df['renko_brick_number'] = brick_numbers
    df['renko_wick_size'] = wick_sizes
    df['renko_brick_end_price'] = brick_end_prices
    df = pd.get_dummies(df, columns=['renko_direction'], drop_first=True)

    return df


#%% Super Guppy
def super_guppy(df):
    fast_lengths = list(range(3, 24, 2))
    slow_lengths = list(range(25, 71, 3))

    for length in fast_lengths:
        df[f'ema_fast_{length}'] = ta.ema(df['close'], length)
    for length in slow_lengths:
        df[f'ema_slow_{length}'] = ta.ema(df['close'], length)

    df['super_guppy'] = df.apply(lambda row: 'up' if all(row[f'ema_fast_{l}'] > row[f'ema_fast_{l+2}'] for l in fast_lengths[:-1]) and \
        all(row[f'ema_slow_{l}'] > row[f'ema_slow_{l+3}'] for l in slow_lengths[:-1]) else ('down' if all(row[f'ema_fast_{l}'] < row[f'ema_fast_{l+2}'] for l in fast_lengths[:-1]) and \
        all(row[f'ema_slow_{l}'] < row[f'ema_slow_{l+3}'] for l in slow_lengths[:-1]) else 'no_trend'), axis=1)

    df = pd.get_dummies(df, columns=['super_guppy'], drop_first=True)

    for length in fast_lengths:
        df.drop(f'ema_fast_{length}', axis=1, inplace=True)
    for length in slow_lengths:
        df.drop(f'ema_slow_{length}', axis=1, inplace=True)
    return df

#%% Monte carlo

import numpy as np
import pandas as pd

def monte_carlo_simulation(df, simulations=500, lookback=500, steps=10, bins=50, smoothing=1.75, granularity=25,
                           drift_length=34, drift_style='Standard', volatility_adj=True, return_style='Log Returns',
                           precision=4, wait_for_steps=False):
    
    def prng(seed, _range=100):
        random = 1.0
        random = ((3.1415926 * random % seed) * 271.828) % _range
        new_seed = ((seed * 1.618) % _range) + random
        return random, new_seed
    
    def returns(end, start, style):
        if style == 'Percent':
            return (end - start) / start
        else:
            return np.log(end / start)
    
    def add_returns(value, returns, style):
        if style == 'Percent':
            return (1.0 + returns) * value
        else:
            return np.exp(returns) * value
    
    def moves(precision, granularity, lookback, drift_style, drift_length, source, return_style, steps, volatility_adj):
        close_prices = df['close']
        open_prices = df['open']
        high_prices = df['high']
        low_prices = df['low']
        
        # ... (implementation of the moves function) ...
        
        return move
    
    def monte_carlo(moves, seed, steps):
        monte = np.zeros(steps)
        over_seed = seed
        polarity_sum = np.sum(moves['polarity'])
        moves_up_sum = np.sum(moves['up'])
        moves_dn_sum = np.sum(moves['down'])
        
        if moves['up'].max() > 0 and moves['down'].max() > 0:
            for i in range(steps):
                polarity_random, polarity_seed = prng(over_seed)
                over_seed = polarity_seed
                if moves['polarity'].probability(0, polarity_sum) >= polarity_random:
                    moves_up_size_1 = len(moves['up']) - 1
                    move_random, move_random_seed = prng(over_seed, moves_up_size_1)
                    over_seed = move_random_seed
                    up_index = int(move_random)
                    move_found = False
                    while not move_found:
                        if up_index > moves_up_size_1:
                            up_index = 0
                        if moves['up'][up_index] == 0:
                            up_index += 1
                            continue
                        move_find_random, move_find_seed = prng(over_seed)
                        over_seed = move_find_seed
                        if moves['up'].probability(up_index, moves_up_sum) >= move_find_random:
                            monte[i] = moves.idx_to_percent(up_index, moves['vol'], True)
                            move_found = True
                            break
                        else:
                            up_index += 1
                else:
                    moves_down_size_1 = len(moves['down']) - 1
                    move_random, move_random_seed = prng(over_seed, moves_down_size_1)
                    over_seed = move_random_seed
                    down_index = int(move_random)
                    move_found = False
                    while not move_found:
                        if down_index > moves_down_size_1:
                            down_index = 0
                        if moves['down'][down_index] == 0:
                            down_index += 1
                            continue
                        move_find_random, move_find_seed = prng(over_seed)
                        over_seed = move_find_seed
                        if moves['down'].probability(down_index, moves_dn_sum) >= move_find_random:
                            monte[i] = moves.idx_to_percent(down_index, moves['vol'], False)
                            move_found = True
                            break
                        else:
                            down_index += 1
        
        return monte, over_seed
    
    def monte_carlo_distribution(moves, seed, bins, steps, sims):
        distribution = np.zeros(bins + 1, dtype=int)
        sim_outcomes = []
        sims_1 = sims - 1
        sum_stdev = 0.0
        reseed = seed
        
        for i in range(sims_1 + 1):
            simulation, seed_sim = sim(moves, reseed, steps)
            reseed = seed_sim
            sum_stdev += np.square(simulation - df['open'])
            sim_outcomes.append(simulation)
        
        sim_lowest = np.min(sim_outcomes)
        sim_highest = np.max(sim_outcomes)
        bin_width = (sim_highest - sim_lowest) / bins
        avg_idx_up = 0.0
        sum_idx_up = 0.0
        avg_idx_dn = 0.0
        sum_idx_dn = 0.0
        
        for i in range(len(sim_outcomes)):
            outcome = sim_outcomes[i]
            idx = int((outcome - sim_lowest) / bin_width)
            if outcome > df['open']:
                avg_idx_up += idx
                sum_idx_up += 1
            else:
                avg_idx_dn += idx
                sum_idx_dn += 1
            distribution[idx] += 1
        
        avg_up = avg_idx_up / sum_idx_up if sum_idx_up > 0 else 0
        avg_dn = avg_idx_dn / sum_idx_dn if sum_idx_dn > 0 else 0
        stdev = np.sqrt(sum_stdev / sims)
        
        return distribution, bin_width, sim_lowest, sim_highest, avg_up, avg_dn, stdev, reseed
    
    def sim(moves, seed, steps):
        simulation = df['open']
        movements, _seed = monte_carlo(moves, seed, steps)
        
        for i in range(steps):
            simulation = add_drift(add_returns(simulation, movements[i], moves['return_style']), moves['drift'], moves['drift_style'])
        
        return simulation, _seed
    
    def add_drift(value, drift, style):
        if style == 'Standard':
            return add_log_returns(value, drift)
        elif style == 'Linear Regression':
            return value + drift
        else:
            return value
    
    def add_log_returns(value, log_returns):
        return np.exp(log_returns) * value
    
    close_prices = df['close']
    open_prices = df['open']
    high_prices = df['high']
    low_prices = df['low']
    
    rand_moves = moves(precision, granularity, lookback, drift_style, drift_length, 'Candle', return_style, steps, volatility_adj)
    seed = 1
    _, re_seed = prng(seed)
    seed = re_seed
    
    distribution, bin_width, lowest_sim, highest_sim, avg_idx_up, avg_idx_down, stdev, _ = monte_carlo_distribution(rand_moves, seed, bins, steps, simulations)
    monte = sinc_filter(distribution, smoothing)
    max_freq = np.max(monte)
    half_bin = bin_width * 0.5
    variation = stdev * 1.0
    dev_up = open_prices + variation
    dev_down = np.maximum(0.0, open_prices - variation)
    avg_up = avg_idx_up * bin_width + lowest_sim
    avg_down = avg_idx_down * bin_width + lowest_sim
    
    # Add the calculated columns to the DataFrame
    df['monte_carlo_high'] = highest_sim
    df['monte_carlo_low'] = lowest_sim
    df['monte_carlo_avg_up'] = avg_up
    df['monte_carlo_avg_down'] = avg_down
    df['monte_carlo_stdev'] = stdev
    
    return df

def sinc_filter(source, length):
    length_1 = length + 1.0
    if length > 0.0 and len(source) > 0:
        source_size = len(source)
        source_size_1 = source_size - 1
        est = np.zeros(source_size)
        
        for i in range(source_size):
            sum_val = 0.0
            sumw = 0.0
            
            for j in range(source_size):
                weight = sinc(i - j, length_1)
                sum_val += weight * source[j]
                sumw += weight
            
            current_price = sum_val / sumw
            est[i] = current_price if current_price >= 0.0 else 0.0
        
        return est
    else:
        return source

def sinc(source, bandwidth):
    omega = np.pi * source / bandwidth
    return np.sin(omega) / omega if source != 0.0 else 1.0



#%% HLT 
import pandas as pd

class HLT:
    def __init__(self, length=23, offset=4, use_high_and_low=False, ema_period=9):
        """
        Ініціалізація класу HLT (Highest-Lowest Trend).

        Параметри:
        length (int): Довжина вікна для визначення найвищих та найнижчих значень. За замовчуванням 23.
        offset (int): Зсув для визначення найвищих та найнижчих значень. За замовчуванням 4.
        use_high_and_low (bool): Якщо True, використовується high та low ціни для визначення тренду. Якщо False, використовується close.
        ema_period (int): Період експоненційної рухомої середньої для гладкості HLT. За замовчуванням 9.
        """
        self.length = length
        self.offset = offset
        self.use_high_and_low = use_high_and_low
        self.ema_period = ema_period
    
    def transform(self, X, y=None):
        # Use high, low or close based on the input flag
        src = X['high'] if self.use_high_and_low else X['close']

        # Calculate highest and lowest values with the specified length and offset
        X['hlt_upper'] = src.rolling(window=self.length).max().shift(self.offset)
        X['hlt_lower'] = src.rolling(window=self.length).min().shift(self.offset)

        # Initialize HLT series
        hlt = pd.Series(index=X.index, dtype='float64')

        for i in range(len(X)):
            if src.iloc[i] > X['hlt_upper'].iloc[i]:
                hlt.iloc[i] = X['hlt_lower'].iloc[i]
            elif src.iloc[i] < X['hlt_lower'].iloc[i]:
                hlt.iloc[i] = X['hlt_upper'].iloc[i]
            else:
                hlt.iloc[i] = hlt.iloc[i-1] if i > 0 else 0.0

        X['hlt'] = np.sign(hlt.diff())
        X['hlt_ema'] = hlt.ewm(span=self.ema_period, adjust=False).mean()

        return X[['hlt','hlt_ema']]

# hlt = HLT()
# df_with_hlt = hlt.transform(X)
# df_with_hlt.hlt = df_with_hlt.hlt * 1000
# X.close.plot()
# df_with_hlt.hlt.plot()
# df_with_hlt.hlt_ema.plot()

#%% Ribbon

import pandas as pd
import numpy as np
import pandas_ta as ta

class Ribbon:
    
    def __init__(self, price_column='close', methods=['phi_smoother', 'tema', 'dema', 'wma', 'ema', 'sma'], lengths=None):
        """
        Ribbon
        Параметри:
        price_column (str): Назва колонки з цінами, яка буде використовуватися для обчислення. За замовчуванням 'close'.
        methods (list): Список методів для обчислення різних середніх. За замовчуванням включає ['phi_smoother', 'tema', 'dema', 'wma', 'ema', 'sma'].
        lengths (list): Список довжин для обчислення середніх. Якщо не вказано, використовується діапазон від 5 до 160 з кроком 5.
        """
        self.price_column = price_column
        self.methods = methods
        self.lengths = lengths
        if self.lengths is None:
            self.lengths = list(range(5, 165, 5))

    def phi_smoother(self, df, column, length, phase=3.7):
        SQRT_PIx2 = np.sqrt(2.0 * np.pi)
        MULTIPLIER = -0.5 / 0.93
        length_2 = length * 0.52353

        if length > 1:
            weights = []
            for i in range(length):
                alpha = (i + phase - length_2) * MULTIPLIER
                beta = 1.0 / (0.2316419 * abs(alpha) + 1.0)
                phi = (np.exp(alpha**2 * -0.5) * -0.398942280) * beta * \
                    (0.319381530 + beta * (-0.356563782 + beta * \
                    (1.781477937 + beta * (-1.821255978 + beta * 1.330274429)))) + 1.011
                if alpha < 0:
                    phi = 1.0 - phi
                weight = phi / SQRT_PIx2
                weights.append(weight)

            weights = np.array(weights)
            if column in df.columns:
                df_smooth = df[column].rolling(window=length).apply(lambda x: np.dot(x[-len(weights):], weights) / np.sum(weights), raw=True)
                return df_smooth
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
        else:
            return df[column]


    def tema(self, series, length):
        return ta.tema(series, length=length)

    def dema(self, series, length):
        return ta.dema(series, length=length)

    def wma(self, series, length):
        return ta.wma(series, length=length)

    def ema(self, series, length):
        return ta.ema(series, length=length)

    def sma(self, series, length):
        return ta.sma(series, length=length)

    def compute_ribbon(self, df, price_column, methods, lengths):
        ribbons = {}
        for method in methods:
            method_function = getattr(self, method)
            if method == 'phi_smoother':
                # Special handling for phi_smoother, which needs the DataFrame and column name
                ribbons[method] = [method_function(df, price_column, length) for length in lengths]
            else:
                # For other methods, assuming they only need the series and length
                ribbons[method] = [method_function(df[price_column], length) for length in lengths]
        return ribbons


    def _transform(self, X, y=None):
        ribbons = self.compute_ribbon(X, self.price_column, self.methods, self.lengths)

        oscillator_values = []
        # Calculating the slope of each ribbon and oscillator values
        for method in self.methods:
            for length in self.lengths:
                ribbon = ribbons[method][self.lengths.index(length)]
                X[f'{method}_{length}_slope'] = ribbon.diff()

            # Calculating ribbon differences
            ribbon_diff = ribbons[method][0] - ribbons[method][-1]
            X[f'{method}_ribbon_diff'] = ribbon_diff
            X[f'{method}_volatility'] = ribbon_diff.rolling(window=30).std()
            oscillator_values.append(ribbon_diff)

        # Simple oscillator: average of differences
        X['composit_trend'] = np.mean(oscillator_values, axis=0)

        return_columns = ['composit_trend'] + [ f'{method}_ribbon_diff' for m in self.methods] + [ f'{method}_volatility' for m in self.methods]
        return X[return_columns]

# # Usage
# ribbon = Ribbon()
# df_with_indicators = ribbon.transform(X)
# df_with_indicators.composit_trend.plot()
# X['close'].plot()