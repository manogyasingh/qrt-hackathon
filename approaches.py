import pandas as pd
import numpy as np
import warnings



def universe_size():
    # Load the universe data
    universe_df = pd.read_parquet("./Learning/universe.parquet")

    # Count how many stocks have value 1 per day
    stocks_per_day = universe_df.apply(lambda row: (row == 1).sum(), axis=1)

    # Display the result
    print(stocks_per_day)

    # Optional: If you want to see the first few rows of the result
    print("\nFirst few counts:")
    print(stocks_per_day.head())

    # Optional: Summary statistics
    print("\nSummary statistics:")
    print(stocks_per_day.describe())

    # Export the stocks per day counts to a CSV file
    stocks_per_day.to_csv("./universe_size.csv")


def count_nulls_in_returns():
    # Load the returns data
    returns_df = pd.read_parquet("./Learning/returns.parquet")
    
    # Count nulls per day (row-wise)
    nulls_per_day = returns_df.isna().sum(axis=1)
    
    # Count nulls per stock (column-wise)
    nulls_per_stock = returns_df.isna().sum(axis=0)
    
    # Export to CSV files
    nulls_per_day.to_csv("./nulls_per_day.csv", header=["null_count"])
    nulls_per_stock.to_csv("./nulls_per_stock.csv", header=["null_count"])
    
    # Optional: Print summary
    print(f"Counted nulls across {len(returns_df)} days and {len(returns_df.columns)} stocks")
    print(f"Average nulls per day: {nulls_per_day.mean():.2f}")
    print(f"Average nulls per stock: {nulls_per_stock.mean():.2f}")
    
    return nulls_per_day, nulls_per_stock


def calculate_return_metrics():
    # Load the returns data
    returns_df = pd.read_parquet("./Learning/returns.parquet")
    
    # Initialize a DataFrame to store metrics
    metrics_df = pd.DataFrame(index=returns_df.columns)
    
    # Calculate metrics for each stock
    metrics_df['mean_return'] = returns_df.mean()
    metrics_df['median_return'] = returns_df.median()
    metrics_df['std_dev'] = returns_df.std()
    metrics_df['sharpe_ratio'] = metrics_df['mean_return'] / metrics_df['std_dev']
    
    # Calculate cumulative return (gross profit)
    metrics_df['gross_profit'] = (1 + returns_df).prod() - 1
    
    # Calculate maximum drawdown
    cum_returns = (1 + returns_df).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    metrics_df['max_drawdown'] = drawdowns.min()
    
    # Calculate win rate (percentage of positive returns)
    metrics_df['win_rate'] = (returns_df > 0).mean()
    
    # Count number of observations (non-null values)
    metrics_df['observations'] = returns_df.count()
    
    # Calculate annualized metrics (assuming 252 trading days per year)
    metrics_df['annualized_return'] = (1 + metrics_df['mean_return']) ** 252 - 1
    metrics_df['annualized_volatility'] = metrics_df['std_dev'] * np.sqrt(252)
    metrics_df['annualized_sharpe'] = metrics_df['annualized_return'] / metrics_df['annualized_volatility']
    
    # Export results to CSV
    metrics_df.to_csv('./return_metrics_by_stock.csv')
    
    # Print summary
    print(f"Calculated return metrics for {len(metrics_df)} stocks")
    print("\nAverage metrics across all stocks:")
    print(metrics_df.mean())
    
    return metrics_df


def countnulls():
    # Load the CSV file containing null counts
    null_counts_df = pd.read_csv('./calculated/nulls_per_stock.csv')

    # Count how many rows have more than 2000 missing data points
    count_above_2000 = (null_counts_df['null_count'] > 2000).sum()

    # Print the result
    print(f"Number of rows with more than 2000 missing data points: {count_above_2000}")


def clean_features_dataset():
    # Load the CSV file containing null counts
    null_counts_df = pd.read_csv('./calculated/nulls_per_stock.csv')
    
    # The CSV file has stock IDs in the first column (without a header)
    # Rename the columns for clarity
    if null_counts_df.columns[0] == 'Unnamed: 0':
        null_counts_df = null_counts_df.rename(columns={'Unnamed: 0': 'stock_id'})
    
    # Identify stocks with more than 2500 missing values
    # We need to use the actual stock IDs from the first column, not the DataFrame index
    stocks_with_high_nulls = null_counts_df[null_counts_df['null_count'] > 2000]
    stocks_to_remove = stocks_with_high_nulls['stock_id'].tolist()
    
    print(f"Found {len(stocks_to_remove)} stocks with more than 2500 missing values")
    print(f"Stocks to be removed: {stocks_to_remove}")
    
    # Load the features data
    print("Loading features data...")
    features_df = pd.read_parquet('./Learning/features.parquet')
    
    print(f"Original features shape: {features_df.shape}")
    
    # Convert stock IDs to strings for matching with column MultiIndex
    stocks_to_remove_str = [str(s) for s in stocks_to_remove]
    
    # Get the feature names (first level of MultiIndex)
    feature_names = features_df.columns.get_level_values(0).unique()
    
    # Create a list of columns to drop
    columns_to_drop = []
    for feature in feature_names:
        for stock in stocks_to_remove_str:
            if (feature, stock) in features_df.columns:
                columns_to_drop.append((feature, stock))
    
    print(f"Removing {len(columns_to_drop)} feature-stock combinations")
    
    # Drop the columns
    features_clean_df = features_df.drop(columns=columns_to_drop)
    
    print(f"Cleaned features shape: {features_clean_df.shape}")
    
    # Save the cleaned dataset
    features_clean_df.to_parquet('./Learning/features-clean.parquet')
    
    print(f"Cleaned dataset saved to './Learning/features-clean.parquet'")
    
    # Verify which stocks remain in the cleaned dataset
    remaining_stocks = set(features_clean_df.columns.get_level_values(1).unique())
    print(f"Number of unique stocks in cleaned dataset: {len(remaining_stocks)}")
    
    # Check if any of the stocks that should have been removed are still present
    problem_stocks = set(stocks_to_remove_str).intersection(remaining_stocks)
    if problem_stocks:
        print(f"WARNING: Found {len(problem_stocks)} problematic stocks that weren't removed: {problem_stocks}")
    else:
        print("All problematic stocks were successfully removed")
    
    return features_clean_df


def count_nans_in_clean_features():
    # Load the cleaned features data
    print("Loading cleaned features data...")
    features_df = pd.read_parquet('./Learning/features-imputed.parquet')
    
    print(f"Cleaned features shape: {features_df.shape}")
    
    # Get unique feature names and stock IDs from the MultiIndex
    feature_names = features_df.columns.get_level_values(0).unique()
    stock_ids = features_df.columns.get_level_values(1).unique()
    
    print(f"Found {len(feature_names)} features and {len(stock_ids)} stocks")
    
    # Create a list to store the NaN count data
    nan_counts = []
    
    # Process each feature-stock combination
    for feature in feature_names:
        print(f"Processing feature {feature}...")
        feature_data = features_df[feature]
        
        # Process stocks in batches to manage memory
        batch_size = 100
        for i in range(0, len(stock_ids), batch_size):
            batch_stocks = stock_ids[i:i+batch_size]
            batch_data = feature_data[batch_stocks]
            
            for stock in batch_stocks:
                try:
                    stock_series = batch_data[stock]
                    nan_count = stock_series.isna().sum()
                    total_count = len(stock_series)
                    nan_percent = (nan_count / total_count) * 100 if total_count > 0 else 0
                    
                    nan_counts.append({
                        'feature': feature,
                        'stock': stock,
                        'nan_count': nan_count,
                        'total_rows': total_count,
                        'nan_percent': nan_percent
                    })
                except Exception as e:
                    print(f"  Error processing feature {feature}, stock {stock}: {e}")
    
    # Convert to DataFrame
    nan_counts_df = pd.DataFrame(nan_counts)
    
    # Save results
    nan_counts_df.to_csv('./Learning/nan_counts_clean_features.csv', index=False)
    print(f"NaN counts saved to './Learning/nan_counts_clean_features.csv'")
    
    # Also create a summary by feature
    feature_summary = nan_counts_df.groupby('feature').agg({
        'nan_count': 'mean',
        'nan_percent': 'mean'
    }).reset_index()
    
    feature_summary.to_csv('./Learning/nan_counts_feature_summary.csv', index=False)
    print(f"Feature NaN summary saved to './Learning/nan_counts_feature_summary.csv'")
    
    # Create a summary by stock
    stock_summary = nan_counts_df.groupby('stock').agg({
        'nan_count': 'mean',
        'nan_percent': 'mean'
    }).reset_index()
    
    stock_summary.to_csv('./Learning/nan_counts_stock_summary.csv', index=False)
    print(f"Stock NaN summary saved to './Learning/nan_counts_stock_summary.csv'")
    
    return nan_counts_df


def impute_missing_values():
    # Load the cleaned features data
    print("Loading cleaned features data...")
    features_df = pd.read_parquet('./Learning/features-clean.parquet')
    
    print(f"Original features shape: {features_df.shape}")
    
    # Get unique feature names and stock IDs from the MultiIndex
    feature_names = features_df.columns.get_level_values(0).unique()
    stock_ids = features_df.columns.get_level_values(1).unique()
    
    print(f"Found {len(feature_names)} features and {len(stock_ids)} stocks")
    print(f"NaN values before imputation: {features_df.isna().sum().sum()}")
    
    # Create a copy to store imputed data
    imputed_df = features_df.copy()
    
    # Process each feature-stock combination
    for feature in feature_names:
        print(f"Imputing feature {feature}...")
        feature_data = imputed_df[feature]
        
        # For each stock, apply imputation
        for stock in stock_ids:
            # Skip if the column doesn't exist
            if stock not in feature_data.columns:
                continue
                
            # Get the stock's data series
            stock_series = feature_data[stock]
            
            # Check for NaN values
            if stock_series.isna().any():
                # Strategy 1: Forward fill (use last known value)
                stock_series_imputed = stock_series.ffill()
                
                # Strategy 2: If still NaNs at the beginning, backfill (use next known value)
                if stock_series_imputed.isna().any():
                    stock_series_imputed = stock_series_imputed.bfill()
                
                # Strategy 3: If still NaNs (unlikely, but possible with all NaN series), 
                # fill with feature mean or zero
                if stock_series_imputed.isna().any():
                    # Calculate mean without NaNs across all stocks for this feature
                    feature_mean = feature_data.mean().mean()
                    if pd.isna(feature_mean):
                        # If mean is still NaN, use zero
                        stock_series_imputed = stock_series_imputed.fillna(0)
                    else:
                        stock_series_imputed = stock_series_imputed.fillna(feature_mean)
                
                # Update the dataframe with imputed values
                imputed_df.loc[:, (feature, stock)] = stock_series_imputed
    
    # Verify no NaN values remain
    remaining_nans = imputed_df.isna().sum().sum()
    print(f"NaN values after imputation: {remaining_nans}")
    
    if remaining_nans > 0:
        # Final pass: replace any remaining NaNs with zeros
        imputed_df = imputed_df.fillna(0)
        print(f"Filled remaining {remaining_nans} NaNs with zeros")
        print(f"Final NaN count: {imputed_df.isna().sum().sum()}")
    
    # Save the imputed dataset
    print("Saving imputed dataset...")
    
    # First save as parquet for efficient storage
    imputed_df.to_parquet('./Learning/features-imputed.parquet')
    
    # Then save as CSV as requested
    imputed_df.to_csv('./Learning/features-imputed.csv')
    
    print("Imputed dataset saved to './Learning/features-imputed.csv' and './Learning/features-imputed.parquet'")
    
    return imputed_df


def printhead():
    import pandas as pd

    # Load the parquet file
    file_path = './Learning/features.parquet'
    features_df = pd.read_parquet(file_path)

    # Print the first 50 rows
    print(features_df.head(50))


def normalise_features():
    import pandas as pd

    unnorm = pd.read_parquet("./Learning/features.parquet")
    print(unnorm.head(10))
    norm = pd.read_parquet("./Learning/normalized_features.parquet")
    print(norm.head(10))


def calculate_feature_return_correlations():
    # Suppress specific numpy warnings about invalid operations
    # These are expected when working with financial data that has many NaN values
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in reduce')
    
    # Load the imputed features data
    print("Loading imputed features data...")
    features_df = pd.read_parquet('./Learning/features-imputed.parquet')
    
    # Load the return metrics data
    print("Loading return metrics data...")
    returns_metrics_df = pd.read_csv('./calculated/return_metrics_by_stock.csv')
    returns_metrics_df['stock'] = returns_metrics_df['stock'].astype(str)  # Convert to string for matching
    returns_metrics_df.set_index('stock', inplace=True)
    
    # Get unique feature names and metrics
    feature_names = features_df.columns.get_level_values(0).unique()
    return_metrics = returns_metrics_df.columns
    
    print(f"Found {len(feature_names)} features and {len(return_metrics)} return metrics to analyze")
    
    # Create a DataFrame to store feature statistics per stock
    feature_stats_df = pd.DataFrame()
    
    # For each feature, calculate statistics per stock with careful NaN handling
    for feature in feature_names:
        print(f"Aggregating statistics for feature {feature}...")
        
        # Get all columns for this feature
        feature_data = features_df[feature]
        
        # Calculate statistics for each stock with proper NaN handling
        for stat_name in ['mean', 'std', 'min', 'max', 'median']:
            col_name = f"{feature}_{stat_name}"
            
            # Use pandas methods that are more robust with NaN values
            if stat_name == 'mean':
                feature_stats_df[col_name] = feature_data.apply(lambda x: pd.Series(x).mean())
            elif stat_name == 'std':
                feature_stats_df[col_name] = feature_data.apply(lambda x: pd.Series(x).std())
            elif stat_name == 'min':
                feature_stats_df[col_name] = feature_data.apply(lambda x: pd.Series(x).min())
            elif stat_name == 'max':
                feature_stats_df[col_name] = feature_data.apply(lambda x: pd.Series(x).max())
            elif stat_name == 'median':
                feature_stats_df[col_name] = feature_data.apply(lambda x: pd.Series(x).median())
    
    # Replace any remaining NaN or Inf values
    feature_stats_df = feature_stats_df.replace([np.inf, -np.inf], np.nan)
    
    # Check for NaN values
    nan_count = feature_stats_df.isna().sum().sum()
    print(f"Found {nan_count} NaN values in feature statistics. Dropping affected rows...")
    
    # Convert index to strings for matching with return metrics
    feature_stats_df.index = feature_stats_df.index.astype(str)
    
    # Join with return metrics
    combined_df = feature_stats_df.join(returns_metrics_df, how='inner')
    print(f"Successfully matched {len(combined_df)} stocks between features and return metrics")
    
    # Calculate correlations with explicit NaN handling
    print("Calculating correlations...")
    correlation_matrix = pd.DataFrame(index=feature_stats_df.columns, columns=return_metrics)
    
    for feature_stat in feature_stats_df.columns:
        for return_metric in return_metrics:
            # Filter out NaN values for this pair
            valid_data = combined_df[[feature_stat, return_metric]].dropna()
            
            # Calculate correlation if enough data and no inf values
            if len(valid_data) > 5 and not np.isinf(valid_data).any().any():
                try:
                    correlation_matrix.loc[feature_stat, return_metric] = valid_data[feature_stat].corr(valid_data[return_metric])
                except Exception as e:
                    print(f"Error calculating correlation between {feature_stat} and {return_metric}: {e}")
                    correlation_matrix.loc[feature_stat, return_metric] = np.nan
            else:
                correlation_matrix.loc[feature_stat, return_metric] = np.nan
    
    # Save the correlation matrix
    correlation_matrix.to_csv('./calculated/feature_return_correlations.csv')
    print("Feature-return correlations saved to './calculated/feature_return_correlations.csv'")
    
    # Find the strongest correlations (with NaN handling)
    flattened = correlation_matrix.stack().reset_index()
    flattened.columns = ['feature_stat', 'return_metric', 'correlation']
    flattened = flattened.dropna()  # Drop rows with NaN correlations
    
    if len(flattened) > 0:
        # Sort by absolute correlation
        flattened['abs_corr'] = flattened['correlation'].abs()
        top_correlations = flattened.sort_values('abs_corr', ascending=False).head(50)
        
        # Save top correlations
        top_correlations.to_csv('./calculated/top_feature_return_correlations.csv', index=False)
        print("Top correlations saved to './calculated/top_feature_return_correlations.csv'")
    else:
        print("No valid correlations found to report")
    
    return correlation_matrix


def calculate_direct_correlations():
    """
    Calculate direct correlations between daily features and returns.
    This analyzes the time-series relationship between features and returns.
    """
    # Suppress specific numpy warnings about invalid operations
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in reduce')
    
    # Load the imputed features data
    print("Loading imputed features data...")
    features_df = pd.read_parquet('./Learning/features-imputed.parquet')
    
    # Load the returns data
    print("Loading returns data...")
    returns_df = pd.read_parquet('./Learning/returns.parquet')
    
    # Get unique feature names
    feature_names = features_df.columns.get_level_values(0).unique()
    
    print(f"Found {len(feature_names)} features to analyze")
    
    # Create a DataFrame to store results
    results = []
    
    # For each feature and stock combination
    total_combinations = 0
    processed_combinations = 0
    
    # Count total combinations for progress tracking
    for feature in feature_names:
        feature_data = features_df[feature]
        total_combinations += len(feature_data.columns)
    
    print(f"Processing {total_combinations} feature-stock combinations...")
    
    # Process each feature
    for feature in feature_names:
        print(f"Processing feature {feature}...")
        
        # Get all columns for this feature
        feature_data = features_df[feature]
        
        # Process each stock
        for stock in feature_data.columns:
            processed_combinations += 1
            if processed_combinations % 100 == 0:
                print(f"Progress: {processed_combinations}/{total_combinations} combinations ({processed_combinations/total_combinations*100:.1f}%)")
            
            # Get the stock's feature data
            stock_feature = feature_data[stock]
            
            # Skip if the stock doesn't exist in returns data
            if str(stock) not in returns_df.columns:
                continue
            
            # Get the stock's returns data
            stock_returns = returns_df[str(stock)]
            
            # Align by date
            aligned_data = pd.DataFrame({
                'feature': stock_feature,
                'return': stock_returns
            })
            
            # Calculate correlations
            # 1. Same-day correlation
            try:
                same_day_corr = aligned_data['feature'].corr(aligned_data['return'])
            except:
                same_day_corr = np.nan
            
            # 2. Feature vs next-day return (predictive correlation)
            try:
                next_day_return = aligned_data['return'].shift(-1)
                next_day_data = pd.DataFrame({
                    'feature': aligned_data['feature'],
                    'next_day_return': next_day_return
                }).dropna()
                
                if len(next_day_data) > 5:
                    predictive_corr = next_day_data['feature'].corr(next_day_data['next_day_return'])
                else:
                    predictive_corr = np.nan
            except:
                predictive_corr = np.nan
            
            # Store results
            results.append({
                'feature': feature,
                'stock': stock,
                'same_day_correlation': same_day_corr,
                'predictive_correlation': predictive_corr
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw correlations
    results_df.to_csv('./calculated/direct_feature_return_correlations.csv', index=False)
    print("Direct correlations saved to './calculated/direct_feature_return_correlations.csv'")
    
    # Calculate average correlation by feature
    feature_summary = results_df.groupby('feature').agg({
        'same_day_correlation': ['mean', 'std', 'count'],
        'predictive_correlation': ['mean', 'std', 'count']
    })
    
    # Flatten the MultiIndex columns
    feature_summary.columns = [f"{col[0]}_{col[1]}" for col in feature_summary.columns]
    feature_summary = feature_summary.reset_index()
    
    # Save feature summary
    feature_summary.to_csv('./calculated/direct_correlation_by_feature.csv', index=False)
    print("Feature correlation summary saved to './calculated/direct_correlation_by_feature.csv'")
    
    # Find top predictive features (based on average absolute correlation)
    feature_summary['abs_pred_corr_mean'] = abs(feature_summary['predictive_correlation_mean'])
    top_predictive = feature_summary.sort_values('abs_pred_corr_mean', ascending=False)
    
    # Save top predictive features
    top_predictive.head(25).to_csv('./calculated/top_predictive_features.csv', index=False)
    print("Top predictive features saved to './calculated/top_predictive_features.csv'")
    
    # Find stocks that are most predictable
    stock_summary = results_df.groupby('stock').agg({
        'predictive_correlation': ['mean', 'std', 'count']
    })
    stock_summary.columns = [f"{col[0]}_{col[1]}" for col in stock_summary.columns]
    stock_summary = stock_summary.reset_index()
    
    # Calculate absolute predictive correlation
    stock_summary['abs_pred_corr'] = abs(stock_summary['predictive_correlation_mean'])
    top_predictable = stock_summary.sort_values('abs_pred_corr', ascending=False)
    
    # Save top predictable stocks
    top_predictable.head(25).to_csv('./calculated/most_predictable_stocks.csv', index=False)
    print("Most predictable stocks saved to './calculated/most_predictable_stocks.csv'")
    
    return results_df, feature_summary, top_predictive, top_predictable


def calculate_multiperiod_correlations():
    """
    Analyze correlations between features and returns across multiple time frames.
    Computes 1-day, 5-day, and 20-day return metrics and their correlations with features.
    """
    # Suppress warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    pd.options.mode.chained_assignment = None
    
    # Load the data
    print("Loading imputed features data...")
    features_df = pd.read_parquet('./Learning/features-imputed.parquet')
    
    print("Loading returns data...")
    returns_df = pd.read_parquet('./Learning/returns.parquet')
    
    # Get unique feature names
    feature_names = features_df.columns.get_level_values(0).unique()
    print(f"Found {len(feature_names)} features to analyze")
    
    # Specify timeframes and metrics
    timeframes = [1, 5, 20]  # days
    metrics = ['return', 'sharpe', 'win_rate', 'max_drawdown']
    
    # Initialize results
    all_results = []
    total_combinations = len(feature_names) * len(returns_df.columns) * len(timeframes) * len(metrics)
    print(f"Calculating correlations for ~{total_combinations} combinations...")
    
    # Process each stock
    stock_count = 0
    for stock_id in returns_df.columns:
        stock_count += 1
        if stock_count % 50 == 0:
            print(f"Processing stock {stock_count}/{len(returns_df.columns)}")
            
        # Get stock returns
        stock_returns = returns_df[stock_id]
        
        # Skip stocks with too few valid returns
        if stock_returns.count() < 30:
            continue
        
        # Calculate multi-period returns and metrics
        return_metrics = pd.DataFrame(index=returns_df.index)
        
        for period in timeframes:
            # Calculate forward returns for different timeframes
            return_metrics[f'return_{period}d'] = stock_returns.rolling(window=period).sum().shift(-period)
            
            # Skip if we don't have enough data for this stock and period
            if return_metrics[f'return_{period}d'].count() < 20:
                continue
                
            # Calculate Sharpe ratio (using daily std dev * sqrt(period))
            return_metrics[f'sharpe_{period}d'] = (
                return_metrics[f'return_{period}d'] / 
                (stock_returns.rolling(window=period).std().shift(-period) * np.sqrt(period))
            )
            
            # Calculate win rate
            return_metrics[f'win_rate_{period}d'] = (
                stock_returns.rolling(window=period).apply(lambda x: (x > 0).mean()).shift(-period)
            )
            
            # Calculate max drawdown
            def max_drawdown(returns):
                cumulative = (1 + returns).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative / peak) - 1
                return drawdown.min()
            
            return_metrics[f'max_drawdown_{period}d'] = (
                stock_returns.rolling(window=period).apply(max_drawdown).shift(-period)
            )
        
        # Process each feature for this stock
        for feature in feature_names:
            # Skip if this feature doesn't exist for this stock
            if stock_id not in features_df[feature].columns:
                continue
                
            # Get feature values
            feature_values = features_df[feature][stock_id]
            
            # Combine feature and return metrics
            combined_data = pd.DataFrame({
                'feature': feature_values
            }).join(return_metrics)
            
            # Calculate correlations for each timeframe and metric
            for period in timeframes:
                for metric in metrics:
                    metric_col = f'{metric}_{period}d'
                    
                    # Skip if this metric wasn't calculated
                    if metric_col not in combined_data.columns:
                        continue
                    
                    # Drop NaN rows
                    valid_data = combined_data[['feature', metric_col]].dropna()
                    
                    # Calculate correlation if we have enough data
                    if len(valid_data) >= 20:
                        try:
                            corr = valid_data['feature'].corr(valid_data[metric_col])
                            
                            # Store the result
                            all_results.append({
                                'feature': feature,
                                'stock': stock_id,
                                'timeframe': period,
                                'metric': metric,
                                'correlation': corr,
                                'abs_correlation': abs(corr),
                                'data_points': len(valid_data)
                            })
                        except Exception as e:
                            # Skip this correlation if there's an error
                            pass
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    results_df.to_csv('./calculated/multiperiod_correlations.csv', index=False)
    print(f"Saved {len(results_df)} correlation results to './calculated/multiperiod_correlations.csv'")
    
    # Create aggregated results by feature
    feature_summary = results_df.groupby(['feature', 'timeframe', 'metric']).agg({
        'correlation': ['mean', 'median', 'std', 'count'],
        'abs_correlation': ['mean', 'median']
    }).reset_index()
    
    # Flatten the column MultiIndex
    feature_summary.columns = [
        '_'.join(col).strip('_') for col in feature_summary.columns.values
    ]
    
    # Save feature summaries
    feature_summary.to_csv('./calculated/feature_timeframe_summary.csv', index=False)
    print("Saved feature summaries to './calculated/feature_timeframe_summary.csv'")
    
    # Create top correlations table
    # Find strongest feature-metric-timeframe combinations
    top_by_metric_timeframe = []
    
    for metric in metrics:
        for period in timeframes:
            # Filter for this metric and timeframe
            subset = results_df[(results_df['metric'] == metric) & (results_df['timeframe'] == period)]
            
            if not subset.empty:
                # Group by feature and calculate average absolute correlation
                feature_avg = subset.groupby('feature')['abs_correlation'].mean().reset_index()
                
                # Get top 5 features
                top_features = feature_avg.nlargest(5, 'abs_correlation')
                
                for _, row in top_features.iterrows():
                    top_by_metric_timeframe.append({
                        'feature': row['feature'],
                        'metric': metric,
                        'timeframe': period,
                        'avg_abs_correlation': row['abs_correlation']
                    })
    
    # Create the final summary table
    top_summary = pd.DataFrame(top_by_metric_timeframe)
    top_summary = top_summary.sort_values(['metric', 'timeframe', 'avg_abs_correlation'], 
                                         ascending=[True, True, False])
    
    # Save the top correlations summary
    top_summary.to_csv('./calculated/top_feature_by_metric_timeframe.csv', index=False)
    print("Saved top features by metric and timeframe to './calculated/top_feature_by_metric_timeframe.csv'")
    
    return results_df, feature_summary, top_summary


calculate_multiperiod_correlations()
