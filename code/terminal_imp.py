import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Load datasets
files = [
    r'D:\B-TECH\SEM_4\h\math\data\ADM.csv', r'D:\B-TECH\SEM_4\h\math\data\AMAZON.csv', r'D:\B-TECH\SEM_4\h\math\data\APPLE.csv', r'D:\B-TECH\SEM_4\h\math\data\CISCO.csv', r'D:\B-TECH\SEM_4\h\math\data\META.csv',
    r'D:\B-TECH\SEM_4\h\math\data\MICROSOFT.csv', r'D:\B-TECH\SEM_4\h\math\data\NETFLIX.csv', r'D:\B-TECH\SEM_4\h\math\data\QUALCOMM.csv', r'D:\B-TECH\SEM_4\h\math\data\STARBUCKS.csv', r'D:\B-TECH\SEM_4\h\math\data\TESLA.csv'
]
stocks = {file.split('\\')[-1].split('.')[0]: pd.read_csv(file) for file in files}

# Preprocess datasets
for stock in stocks.values():
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock.set_index('Date', inplace=True)
    stock['Close'] = stock['Close'].str.replace('$', '', regex=False).astype(float)  # Remove dollar symbol and convert to float
    stock.sort_index(inplace=True)  # Ensure the DateTime index is sorted

# Function to calculate returns
def calculate_returns(stock_data, stock_name):
    if 'Close' in stock_data.columns:
        return stock_data['Close'].pct_change().dropna()
    else:
        print(f"'Close' column not found in {stock_name}. Columns present: {stock_data.columns}")
        return pd.Series()

# Calculate returns for each stock
returns = {name: calculate_returns(data, name) for name, data in stocks.items()}

# Remove empty returns series
returns = {name: ret for name, ret in returns.items() if not ret.empty}

# Portfolio optimization using proximal gradient method
def optimize_portfolio(returns):
    returns_df = pd.DataFrame(returns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    def portfolio_volatility(weights, mean_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(portfolio_volatility, num_assets * [1. / num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def get_optimal_portfolio(returns, initial_investment):
    result = optimize_portfolio(returns)
    optimal_weights = result.x
    portfolio = {name: (weight * initial_investment, weight) for name, weight in zip(returns.keys(), optimal_weights)}
    return portfolio

def get_optimal_portfolio_selected(stocks_selected, initial_investment):
    selected_returns = {name: returns[name] for name in stocks_selected if name in returns}
    return get_optimal_portfolio(selected_returns, initial_investment)

# Worst performing stocks in a date range
def get_worst_performing_stocks(start_date, end_date):
    date_range_returns = {name: data.loc[start_date:end_date]['Close'].pct_change().dropna() for name, data in stocks.items() if 'Close' in data.columns}
    mean_returns = {name: data.mean() for name, data in date_range_returns.items()}
    sorted_stocks = sorted(mean_returns.items(), key=lambda x: x[1])
    return sorted_stocks

# Visualization of performance metrics2
def visualize_performance():
    returns_df = pd.DataFrame(returns)
    sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Stock Returns')
    plt.show()

# Predicted returns for a specific stock using proximal gradient
def predict_stock_returns(stock_name):
    if stock_name in stocks:
        stock_data = stocks[stock_name]
        stock_data['Returns'] = stock_data['Close'].pct_change().dropna()
        mean_return = stock_data['Returns'].mean()
        cov_return = stock_data['Returns'].cov(stock_data['Returns'])
        return mean_return, cov_return
    else:
        print(f"Stock {stock_name} not found.")
        return None, None

# Performance of a specific stock over a date range
def visualize_stock_performance(stock_name, start_date, end_date):
    if stock_name in stocks:
        stock_data = stocks[stock_name]
        filtered_data = stock_data.loc[start_date:end_date]
        plt.plot(filtered_data.index, filtered_data['Close'])
        plt.title(f'Performance of {stock_name} from {start_date} to {end_date}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.show()
    else:
        print(f"Stock {stock_name} not found.")

# Performance of selected stocks over the past month
def visualize_selected_stocks_performance(stocks_selected):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    for stock_name in stocks_selected:
        if stock_name in stocks:
            stock_data = stocks[stock_name]
            filtered_data = stock_data.loc[start_date:end_date]
            plt.plot(filtered_data.index, filtered_data['Close'], label=stock_name)
        else:
            print(f"Stock {stock_name} not found.")
    plt.title(f'Performance of selected stocks from {start_date.date()} to {end_date.date()}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Main function to interact with the user
def main():
    while True:
        print("Select an option:")
        print("1. Optimize Investment Portfolio")
        print("2. Analyze Worst Performing Stocks")
        print("3. Visualize Performance Metrics")
        print("4. Predict Returns for a Specific Stock")
        print("5. Visualize Performance of a Specific Stock")
        print("6. Visualize Performance of Selected Stocks for the Past Month")
        print("7. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            sub_choice = input("Choose data source:\n1. Current stocks\n2. Past stocks\n3. Selected stocks\nEnter choice: ")
            initial_investment = float(input("Enter the amount you want to invest: "))
            if sub_choice == '1':
                portfolio = get_optimal_portfolio(returns, initial_investment)
                print("Optimized Portfolio:")
                for stock, (amount, weight) in portfolio.items():
                    print(f"{stock}: ${amount:.2f} ({weight * 100:.2f}%)")
            elif sub_choice == '2':
                past_date = input("Enter the date for past stock analysis (YYYY-MM-DD): ")
                past_returns = {name: data.loc[:past_date]['Close'].pct_change().dropna() for name, data in stocks.items() if 'Close' in data.columns and past_date in data.index}
                past_returns = {name: ret for name, ret in past_returns.items() if not ret.empty}
                portfolio = get_optimal_portfolio(past_returns, initial_investment)
                print(f"Optimized Portfolio for {past_date}:")
                for stock, (amount, weight) in portfolio.items():
                    print(f"{stock}: ${amount:.2f} ({weight * 100:.2f}%)")
            elif sub_choice == '3':
                selected_stocks = input("Enter the stock names separated by commas: ").split(',')
                selected_stocks = [stock.strip() for stock in selected_stocks]
                portfolio = get_optimal_portfolio_selected(selected_stocks, initial_investment)
                print("Optimized Portfolio for selected stocks:")
                for stock, (amount, weight) in portfolio.items():
                    print(f"{stock}: ${amount:.2f} ({weight * 100:.2f}%)")
        elif choice == '2':
            start_date = input("Enter the start date (YYYY-MM-DD): ")
            end_date = input("Enter the end date (YYYY-MM-DD): ")
            worst_performing_stocks = get_worst_performing_stocks(start_date, end_date)
            print("Worst Performing Stocks:")
            for stock, return_val in worst_performing_stocks:
                print(f"{stock}: {return_val * 100:.2f}%")
        elif choice == '3':
            visualize_performance()
        elif choice == '4':
            stock_name = input("Enter the stock name: ")
            mean_return, cov_return = predict_stock_returns(stock_name)
            if mean_return is not None:
                print(f"Predicted mean return for {stock_name}: {mean_return * 100:.2f}%")
                print(f"Predicted return variance for {stock_name}: {cov_return:.4f}")
        elif choice == '5':
            stock_name = input("Enter the stock name: ")
            start_date = input("Enter the start date (YYYY-MM-DD): ")
            end_date = input("Enter the end date (YYYY-MM-DD): ")
            visualize_stock_performance(stock_name, start_date, end_date)
        elif choice == '6':
            selected_stocks = input("Enter the stock names separated by commas: ").split(',')
            selected_stocks = [stock.strip() for stock in selected_stocks]
            visualize_selected_stocks_performance(selected_stocks)
        elif choice == '7':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
