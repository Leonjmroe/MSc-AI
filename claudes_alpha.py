import pandas as pd
import numpy as np

def calculate_alpha(df):
    """
    Novel trading strategy combining volume distribution analysis with price momentum.
    
    Parameters:
    df: DataFrame with columns ['timestamp', 'close', 'volume']
    
    Returns:
    DataFrame with signals
    """
    # Calculate base metrics
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close']).diff()
    
    # Volume analysis
    df['vol_ma'] = df['volume'].rolling(window=12).mean()  # 1-hour MA
    df['vol_std'] = df['volume'].rolling(window=12).std()
    df['vol_z_score'] = (df['volume'] - df['vol_ma']) / df['vol_std']
    
    # Volume distribution features
    df['vol_skew'] = df['volume'].rolling(window=24).skew()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    # Price momentum features
    df['mom_12'] = df['log_returns'].rolling(window=12).sum()
    df['mom_24'] = df['log_returns'].rolling(window=24).sum()
    
    # Novel volume-weighted momentum
    df['vol_weighted_mom'] = df['mom_12'] * np.sqrt(df['vol_ratio'])
    
    # Innovative mean reversion component
    df['price_deviation'] = (df['close'] - df['close'].rolling(window=24).mean()) / \
                           df['close'].rolling(window=24).std()
    
    # Generate alpha signal
    df['alpha'] = (
        # Volume distribution component
        0.4 * (df['vol_z_score'].clip(-2, 2) / 2) +
        # Momentum component
        0.3 * (df['vol_weighted_mom'] / df['vol_weighted_mom'].rolling(window=24).std()) +
        # Mean reversion component
        -0.3 * df['price_deviation'].clip(-2, 2) / 2
    )
    
    # Generate trading signals
    df['signal'] = 0
    signal_threshold = 0.8
    df.loc[df['alpha'] > signal_threshold, 'signal'] = 1
    df.loc[df['alpha'] < -signal_threshold, 'signal'] = -1
    
    # Risk management
    df['volatility'] = df['returns'].rolling(window=24).std()
    df.loc[df['volatility'] > df['volatility'].quantile(0.95), 'signal'] = 0
    
    return df

def calculate_position_sizes(df):
    """
    Calculate position sizes based on volatility and signal strength
    """
    df['position_size'] = df['signal'] * (1 - df['volatility'] / df['volatility'].max())
    return df

def backtest_strategy(df):
    """
    Backtest the strategy with transaction costs
    """
    transaction_cost = 0.001  # 1 bps per trade
    df['position_change'] = df['position_size'].diff().abs()
    df['transaction_costs'] = df['position_change'] * transaction_cost
    df['strategy_returns'] = df['position_size'].shift() * df['returns'] - df['transaction_costs']
    return df



import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def calculate_advanced_alpha(df):
    """
    Enhanced strategy focusing on institutional order flow patterns
    """
    # Basic price and volume metrics
    df['log_returns'] = np.log(df['close']).diff()
    df['volume_returns'] = df['volume'].pct_change()
    
    # Advanced volume analysis
    window_sizes = [12, 24, 48]  # 1h, 2h, 4h windows
    
    for window in window_sizes:
        # Volume distribution analysis
        df[f'vol_skew_{window}'] = df['volume'].rolling(window=window).apply(skew)
        df[f'vol_kurt_{window}'] = df['volume'].rolling(window=window).apply(kurtosis)
        
        # Price impact analysis
        df[f'price_impact_{window}'] = (df['log_returns'].abs() / 
                                      np.log1p(df['volume'])).rolling(window).mean()
        
        # Volume/return relationship
        df[f'vol_ret_corr_{window}'] = (df['volume_returns'].rolling(window)
                                       .corr(df['log_returns'].abs()))
    
    # Institutional flow indicators
    df['institutional_accumulation'] = (
        # High volume with low price impact suggests institutional accumulation
        (df['volume'] > df['volume'].rolling(48).mean()) &
        (df['price_impact_48'] < df['price_impact_48'].rolling(48).quantile(0.3))
    ).astype(int)
    
    # VWAP deviation analysis
    df['vwap'] = (df['close'] * df['volume']).rolling(48).sum() / df['volume'].rolling(48).sum()
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['close'].rolling(48).std()
    
    # Adaptive timeframe selection
    df['market_activity'] = df['volume'] / df['volume'].rolling(48).mean()
    df['optimal_window'] = np.select(
        [df['market_activity'] > 2, df['market_activity'] > 1.5],
        [12, 24],
        default=48
    )
    
    # Dynamic alpha calculation
    df['alpha'] = (
        # Volume distribution component
        0.3 * (df['vol_skew_24'] / df['vol_skew_24'].rolling(48).std()) +
        
        # Price impact component (inverse)
        -0.3 * (df['price_impact_24'] / df['price_impact_24'].rolling(48).mean()) +
        
        # VWAP deviation component
        -0.2 * df['vwap_dev'] +
        
        # Institutional accumulation component
        0.2 * (df['institutional_accumulation'] - 
               df['institutional_accumulation'].rolling(24).mean())
    )
    
    # Generate trading signals with dynamic thresholds
    df['signal'] = 0
    df['threshold'] = df['alpha'].rolling(48).std() * 1.5
    
    df.loc[df['alpha'] > df['threshold'], 'signal'] = 1
    df.loc[df['alpha'] < -df['threshold'], 'signal'] = -1
    
    # Risk management
    df['volatility'] = df['log_returns'].rolling(window=24).std()
    df['volume_toxicity'] = df['vol_kurt_24'].rolling(24).mean()
    
    # Reduce positions when market conditions are unfavorable
    risk_conditions = (
        (df['volatility'] > df['volatility'].rolling(48).quantile(0.9)) |
        (df['volume_toxicity'] > df['volume_toxicity'].rolling(48).quantile(0.9))
    )
    df.loc[risk_conditions, 'signal'] = df.loc[risk_conditions, 'signal'] * 0.5
    
    return df

def calculate_adaptive_positions(df):
    """
    Position sizing based on market conditions and signal conviction
    """
    # Base position size on signal strength
    df['position_size'] = df['signal'] * (df['alpha'].abs() / df['alpha'].rolling(48).max())
    
    # Adjust for market conditions
    df['market_quality'] = (
        (1 - df['volatility'] / df['volatility'].rolling(48).max()) *
        (1 - df['volume_toxicity'] / df['volume_toxicity'].rolling(48).max())
    )
    
    df['final_position'] = df['position_size'] * df['market_quality']
    return df



import pandas as pd
import numpy as np
from scipy.stats import linregress

def detect_order_blocks(df):
    """
    Detect potential order blocks using price action and CVD analysis
    """
    # Calculate returns and volatility
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close']).diff()
    df['volatility'] = df['returns'].rolling(24).std()
    
    # Detect explosive price moves
    df['return_zscore'] = (df['returns'] - df['returns'].rolling(48).mean()) / \
                         df['returns'].rolling(48).std()
    
    # CVD analysis
    df['cvd_change'] = df['cvd'].diff()
    df['cvd_acceleration'] = df['cvd_change'].diff()
    df['cvd_power'] = df['cvd_change'].rolling(12).sum() * \
                     df['cvd_change'].rolling(12).std()
    
    # Order block detection
    window = 12  # 1-hour window
    df['price_range'] = df['high'] - df['low']
    df['avg_range'] = df['price_range'].rolling(48).mean()
    
    # Identify potential order blocks
    df['is_order_block'] = 0
    
    # Conditions for order block formation
    order_block_conditions = (
        # Explosive move
        (df['return_zscore'].abs() > 2.5) &
        # Significant CVD accumulation
        (df['cvd_power'].abs() > df['cvd_power'].rolling(48).quantile(0.9)) &
        # Compressed price range before explosion
        (df['price_range'].shift(1) < df['avg_range'] * 0.7) &
        # High volume relative to recent history
        (df['volume'] > df['volume'].rolling(48).mean() * 1.5)
    )
    
    df.loc[order_block_conditions, 'is_order_block'] = 1
    
    # Calculate order block levels
    df['ob_high'] = np.nan
    df['ob_low'] = np.nan
    
    for idx in df[df['is_order_block'] == 1].index:
        df.loc[idx, 'ob_high'] = max(df.loc[idx:idx+window, 'high'])
        df.loc[idx, 'ob_low'] = min(df.loc[idx:idx+window, 'low'])
    
    return df

def calculate_order_block_alpha(df):
    """
    Generate trading signals based on order blocks and CVD
    """
    # Detect order blocks
    df = detect_order_blocks(df)
    
    # CVD trend analysis
    df['cvd_trend'] = df['cvd'].rolling(24).apply(
        lambda x: linregress(np.arange(len(x)), x)[0]
    )
    
    # Order block interaction
    df['distance_to_nearest_ob'] = np.nan
    df['ob_interaction'] = 0
    
    # Find distance to nearest order block levels
    for i in range(len(df)):
        if i < 48:  # Skip first periods
            continue
            
        # Look back for recent order blocks
        recent_obs = df[df['is_order_block'] == 1].iloc[max(0, i-96):i]
        
        if len(recent_obs) > 0:
            distances_high = recent_obs['ob_high'] - df['close'].iloc[i]
            distances_low = df['close'].iloc[i] - recent_obs['ob_low']
            
            # Find closest level
            min_distance = min(
                abs(distances_high.min()) if len(distances_high) > 0 else float('inf'),
                abs(distances_low.min()) if len(distances_low) > 0 else float('inf')
            )
            
            df.loc[df.index[i], 'distance_to_nearest_ob'] = min_distance
            
            # Detect interaction with order block levels
            if min_distance < df['avg_range'].iloc[i] * 0.5:
                df.loc[df.index[i], 'ob_interaction'] = 1
    
    # Generate alpha signal
    df['alpha'] = (
        # CVD momentum
        0.4 * (df['cvd_trend'] / df['cvd_trend'].rolling(48).std()) +
        
        # Order block proximity effect
        0.3 * (-df['distance_to_nearest_ob'] / df['avg_range']) +
        
        # CVD power
        0.3 * (df['cvd_power'] / df['cvd_power'].rolling(48).std())
    )
    
    # Generate trading signals
    df['signal'] = 0
    
    # Long signals
    long_conditions = (
        (df['alpha'] > df['alpha'].rolling(48).std() * 1.5) &
        (df['cvd_trend'] > 0) &
        (df['ob_interaction'] == 1)
    )
    
    # Short signals
    short_conditions = (
        (df['alpha'] < -df['alpha'].rolling(48).std() * 1.5) &
        (df['cvd_trend'] < 0) &
        (df['ob_interaction'] == 1)
    )
    
    df.loc[long_conditions, 'signal'] = 1
    df.loc[short_conditions, 'signal'] = -1
    
    # Risk management
    df['position_size'] = df['signal'] * (
        1 - (df['distance_to_nearest_ob'] / df['distance_to_nearest_ob'].rolling(48).max())
    )
    
    return df




import pandas as pd
import numpy as np
from scipy.stats import linregress, skew, kurtosis
from scipy.signal import find_peaks

class MarketPatternDetector:
    """
    Advanced detection of market patterns including order blocks, 
    liquidity voids, and institutional footprints
    """
    def detect_order_blocks_ml(self, df):
        """
        Machine learning inspired approach to order block detection
        """
        # Feature creation
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_to_range'] = df['body_size'] / (df['high'] - df['low'])
        
        # Momentum features
        for window in [12, 24, 48]:
            # Price momentum
            df[f'mom_{window}'] = df['close'].pct_change(window)
            # Volume momentum
            df[f'vol_mom_{window}'] = df['volume'].pct_change(window)
            # CVD momentum
            df[f'cvd_mom_{window}'] = df['cvd'].diff(window)
        
        # Detect compression patterns
        df['compression'] = (
            (df['high'] - df['low']).rolling(12).std() /
            (df['high'] - df['low']).rolling(48).std()
        )
        
        # Detect explosions from compression
        df['explosion'] = (
            (df['high'].rolling(3).max() - df['low'].rolling(3).min()) /
            (df['high'].rolling(12).std())
        )
        
        # Smart order block detection
        df['ob_score'] = (
            # Price compression followed by expansion
            (df['compression'] < 0.5).astype(int) * 
            (df['explosion'].shift(-1) > 2).astype(int) *
            # Volume and CVD confirmation
            (df['volume'] > df['volume'].rolling(48).mean() * 1.2).astype(int) *
            (abs(df['cvd_mom_12']) > df['cvd_mom_12'].rolling(48).std() * 2).astype(int)
        )
        
        return df

    def detect_liquidity_voids(self, df):
        """
        Detect areas of potential liquidity voids
        """
        # Volume profile analysis
        df['vol_profile'] = df['volume'].rolling(48).apply(
            lambda x: np.histogram(x, bins=10)[0]
        )
        
        # Detect low liquidity zones
        df['liquidity_score'] = (
            df['volume'].rolling(12).mean() / 
            df['volume'].rolling(48).mean()
        )
        
        # Find peaks in price movement
        prices = df['close'].values
        peaks, _ = find_peaks(prices, distance=12)
        troughs, _ = find_peaks(-prices, distance=12)
        
        # Mark potential liquidity voids
        df['liquidity_void'] = 0
        df.loc[peaks, 'liquidity_void'] = 1
        df.loc[troughs, 'liquidity_void'] = -1
        
        return df

    def analyze_cvd_patterns(self, df):
        """
        Advanced CVD pattern analysis
        """
        # CVD distribution analysis
        df['cvd_zscore'] = (
            (df['cvd'] - df['cvd'].rolling(48).mean()) /
            df['cvd'].rolling(48).std()
        )
        
        # CVD acceleration
        df['cvd_acceleration'] = df['cvd'].diff().diff()
        
        # CVD divergence with price
        df['cvd_price_divergence'] = (
            df['cvd'].rolling(24).corr(df['close'])
        )
        
        # Detect CVD accumulation/distribution
        df['cvd_accumulation'] = (
            (df['cvd'].rolling(24).sum() > 0) &
            (df['volume'] > df['volume'].rolling(48).mean()) &
            (df['close'] > df['close'].shift(24))
        ).astype(int)
        
        # CVD trend strength
        df['cvd_trend_strength'] = abs(
            df['cvd'].rolling(24).apply(
                lambda x: linregress(np.arange(len(x)), x)[0]
            )
        )
        
        return df

    def detect_institutional_patterns(self, df):
        """
        Detect sophisticated institutional trading patterns
        """
        # Iceberg order detection
        df['consistent_buying'] = (
            (df['cvd'].rolling(12).sum() > 0) &
            (df['close'].rolling(12).std() < df['close'].rolling(48).std()) &
            (df['volume'] > df['volume'].rolling(48).mean() * 1.2)
        ).astype(int)
        
        # Smart money divergence
        df['smart_money_divergence'] = (
            (df['cvd'].rolling(24).sum() * df['close'].pct_change(24)) < 0
        ).astype(int)
        
        # Institutional absorption
        df['absorption'] = (
            (df['volume'] > df['volume'].rolling(48).mean() * 1.5) &
            (abs(df['close'].pct_change()) < df['close'].pct_change().rolling(48).std()) &
            (abs(df['cvd']) > df['cvd'].rolling(48).std() * 2)
        ).astype(int)
        
        return df

def generate_alpha_signals(df):
    """
    Generate trading signals based on all detected patterns
    """
    detector = MarketPatternDetector()
    
    # Apply all detection methods
    df = detector.detect_order_blocks_ml(df)
    df = detector.detect_liquidity_voids(df)
    df = detector.analyze_cvd_patterns(df)
    df = detector.detect_institutional_patterns(df)
    
    # Combined alpha signal
    df['alpha'] = (
        # Order block component
        0.3 * df['ob_score'] +
        # Liquidity void component
        0.2 * df['liquidity_void'] * (1 - df['liquidity_score']) +
        # CVD component
        0.3 * df['cvd_zscore'] +
        # Institutional patterns component
        0.2 * (df['absorption'] - df['smart_money_divergence'])
    )
    
    # Generate trading signals
    df['signal'] = np.select(
        [
            (df['alpha'] > 1.5) & (df['cvd_trend_strength'] > 0),
            (df['alpha'] < -1.5) & (df['cvd_trend_strength'] > 0)
        ],
        [1, -1],
        default=0
    )
    
    return df


import pandas as pd
import numpy as np
from scipy.stats import linregress, skew
from scipy.signal import argrelextrema

class AdvancedMarketPatterns:
    def analyze_cross_timeframe_momentum(self, df, timeframes=[5, 15, 30, 60]):
        """
        Detect divergences across multiple timeframes
        """
        base_minutes = 5  # Assuming df is 5-minute data
        
        # Resample data to different timeframes
        timeframe_data = {}
        for tf in timeframes:
            # Resample factor
            factor = tf // base_minutes
            resampled = df.copy()
            
            # Calculate momentum metrics for each timeframe
            resampled['momentum'] = resampled['close'].diff(factor)
            resampled['volume_momentum'] = resampled['volume'].rolling(factor).sum().diff()
            resampled['cvd_momentum'] = resampled['cvd'].diff(factor)
            
            # RSI for each timeframe
            delta = resampled['close'].diff(factor)
            gain = (delta.where(delta > 0, 0)).rolling(factor*2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(factor*2).mean()
            rs = gain / loss
            resampled['rsi'] = 100 - (100 / (1 + rs))
            
            timeframe_data[tf] = resampled
        
        # Detect divergences
        df['momentum_divergence'] = 0
        for i in range(len(timeframes)-1):
            tf1, tf2 = timeframes[i], timeframes[i+1]
            
            # Price momentum divergence
            price_div = (
                np.sign(timeframe_data[tf1]['momentum']) !=
                np.sign(timeframe_data[tf2]['momentum'])
            )
            
            # RSI divergence
            rsi_div = (
                np.sign(timeframe_data[tf1]['rsi'].diff()) !=
                np.sign(timeframe_data[tf2]['rsi'].diff())
            )
            
            df.loc[price_div & rsi_div, 'momentum_divergence'] += 1
        
        return df

    def detect_market_maker_traps(self, df):
        """
        Identify potential market maker trap patterns
        """
        # Detect stop runs
        df['high_breakout'] = (
            (df['high'] > df['high'].rolling(48).max().shift(1)) &
            (df['close'] < df['open']) &
            (df['volume'] > df['volume'].rolling(48).mean() * 1.5)
        )
        
        df['low_breakout'] = (
            (df['low'] < df['low'].rolling(48).min().shift(1)) &
            (df['close'] > df['open']) &
            (df['volume'] > df['volume'].rolling(48).mean() * 1.5)
        )
        
        # Detect liquidity sweeps
        df['high_sweep'] = (
            df['high_breakout'] &
            (df['cvd'] < df['cvd'].rolling(12).min().shift(1))
        )
        
        df['low_sweep'] = (
            df['low_breakout'] &
            (df['cvd'] > df['cvd'].rolling(12).max().shift(1))
        )
        
        # Identify trap setups
        df['trap_setup'] = 0
        
        # High trap (false breakout above resistance)
        high_trap_conditions = (
            df['high_sweep'] &
            (df['close'] < df['open']) &
            (df['cvd'].diff() < 0)
        )
        
        # Low trap (false breakout below support)
        low_trap_conditions = (
            df['low_sweep'] &
            (df['close'] > df['open']) &
            (df['cvd'].diff() > 0)
        )
        
        df.loc[high_trap_conditions, 'trap_setup'] = -1  # Short signal
        df.loc[low_trap_conditions, 'trap_setup'] = 1    # Long signal
        
        # Trap confirmation using volume and CVD
        df['trap_conviction'] = 0
        
        for idx in df[df['trap_setup'] != 0].index:
            forward_cvd = df['cvd'].loc[idx:idx+12].diff().sum()
            forward_volume = df['volume'].loc[idx:idx+12].sum()
            
            if (abs(forward_cvd) > df['cvd'].diff().abs().rolling(48).mean().loc[idx] * 2 and
                forward_volume > df['volume'].rolling(48).mean().loc[idx] * 1.5):
                df.loc[idx, 'trap_conviction'] = df.loc[idx, 'trap_setup']
        
        return df

    def detect_algo_patterns(self, df):
        """
        Detect and analyze algorithmic trading patterns
        """
        # VWAP calculation and analysis
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # TWAP calculation (theoretical)
        df['twap'] = df['close'].expanding().mean()
        
        # Detect potential algo execution
        df['algo_footprint'] = 0
        
        # VWAP trading detection
        vwap_conditions = (
            (abs(df['close'] - df['vwap']) < df['close'].rolling(48).std() * 0.2) &
            (df['volume'] > df['volume'].rolling(48).mean()) &
            (df['cvd'].diff().abs() < df['cvd'].diff().abs().rolling(48).mean())
        )
        
        # TWAP trading detection
        twap_conditions = (
            (df['volume'].rolling(12).std() / df['volume'].rolling(12).mean() < 0.3) &
            (df['close'].rolling(12).std() / df['close'].rolling(12).mean() < 0.001)
        )
        
        # Iceberg order detection
        iceberg_conditions = (
            (df['volume'] > df['volume'].rolling(48).mean() * 1.2) &
            (df['close'].rolling(12).std() < df['close'].rolling(48).std() * 0.5) &
            (abs(df['cvd'].diff()) > df['cvd'].diff().abs().rolling(48).mean() * 1.5)
        )
        
        # Score different algo patterns
        df.loc[vwap_conditions, 'algo_footprint'] += 1
        df.loc[twap_conditions, 'algo_footprint'] += 1
        df.loc[iceberg_conditions, 'algo_footprint'] += 1
        
        # Analyze algo impact
        df['algo_impact'] = 0
        
        for idx in df[df['algo_footprint'] > 1].index:
            # Look forward to see price impact
            forward_return = df['close'].loc[idx:idx+24].pct_change().sum()
            forward_vol = df['volume'].loc[idx:idx+24].sum()
            
            # Score the impact
            impact_score = forward_return * np.sign(df['cvd'].diff().loc[idx]) * \
                          (forward_vol / df['volume'].rolling(48).mean().loc[idx])
            
            df.loc[idx, 'algo_impact'] = impact_score
        
        return df

def generate_combined_signals(df):
    """
    Generate trading signals combining all pattern types
    """
    analyzer = AdvancedMarketPatterns()
    
    # Apply all analyses
    df = analyzer.analyze_cross_timeframe_momentum(df)
    df = analyzer.detect_market_maker_traps(df)
    df = analyzer.detect_algo_patterns(df)
    
    # Generate combined alpha signal
    df['alpha'] = (
        # Cross-timeframe momentum component
        0.3 * df['momentum_divergence'] * np.sign(df['cvd'].diff()) +
        
        # Market maker trap component
        0.4 * df['trap_conviction'] * (1 + abs(df['cvd'].diff()) / 
                                     df['cvd'].diff().abs().rolling(48).mean()) +
        
        # Algorithmic trading component
        0.3 * df['algo_impact'] * (df['algo_footprint'] / 3)
    )
    
    # Generate final trading signals
    df['signal'] = np.select(
        [
            (df['alpha'] > 1.5) & (df['cvd'].diff() > 0),
            (df['alpha'] < -1.5) & (df['cvd'].diff() < 0)
        ],
        [1, -1],
        default=0
    )
    
    return df