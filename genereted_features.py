import pandas as pd
import pandas_ta as ta
import numpy as np

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


def super_guppy(df):
    fast_lengths = list(range(3, 24, 2))
    slow_lengths = list(range(25, 71, 3))

    for length in fast_lengths:
        df[f'ema_fast_{length}'] = ta.ema(df['close'], length)
    for length in slow_lengths:
        df[f'ema_slow_{length}'] = ta.ema(df['close'], length)

    # Використання lambda функції для перевірки умов в кожному рядку
    df['super_guppy'] = df.apply(lambda row: 'up' if all(row[f'ema_fast_{l}'] > row[f'ema_fast_{l+2}'] for l in fast_lengths[:-1]) and \
        all(row[f'ema_slow_{l}'] > row[f'ema_slow_{l+3}'] for l in slow_lengths[:-1]) else ('down' if all(row[f'ema_fast_{l}'] < row[f'ema_fast_{l+2}'] for l in fast_lengths[:-1]) and \
        all(row[f'ema_slow_{l}'] < row[f'ema_slow_{l+3}'] for l in slow_lengths[:-1]) else 'no_trend'), axis=1)

    df = pd.get_dummies(df, columns=['super_guppy'], drop_first=True)

    # for length in fast_lengths:
    #     df.drop(f'ema_fast_{length}', axis=1, inplace=True)
    # for length in slow_lengths:
    #     df.drop(f'ema_slow_{length}', axis=1, inplace=True)

    return df
