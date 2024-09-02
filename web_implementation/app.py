import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64

app = Flask(__name__)

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

def proximal_grading_optimal(returns, initial_investment):
    result = optimize_portfolio(returns)
    optimal_weights = result.x
    portfolio = {name: (weight * initial_investment, weight) for name, weight in zip(returns.keys(), optimal_weights)}
    return portfolio

def proximal_grading_optimal_selected(stocks_selected, initial_investment):
    selected_returns = {name: returns[name] for name in stocks_selected if name in returns}
    return proximal_grading_optimal(selected_returns, initial_investment)

# Worst performing stocks in a date range
def get_worst_performing_stocks(start_date, end_date):
    date_range_returns = {name: data.loc[start_date:end_date]['Close'].pct_change().dropna() for name, data in stocks.items() if 'Close' in data.columns}
    mean_returns = {name: data.mean() for name, data in date_range_returns.items()}
    sorted_stocks = sorted(mean_returns.items(), key=lambda x: x[1])
    return sorted_stocks

# Visualization of performance metrics
def visualize_performancXGBooste():
    returns_df = pd.DataFrame(returns)
    sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Stock Returns')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

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
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    else:
        print(f"Stock {stock_name} not found.")
        return None

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
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize_portfolio', methods=['GET', 'POST'])
def optimize_portfolio_route():
    if request.method == 'POST':
        sub_choice = request.form['sub_choice']
        initial_investment = float(request.form['initial_investment'])
        if sub_choice == '1':
            portfolio = proximal_grading_optimal(returns, initial_investment)
        elif sub_choice == '2':
            past_date = request.form['past_date']
            past_returns = {name: data.loc[:past_date]['Close'].pct_change().dropna() for name, data in stocks.items() if 'Close' in data.columns and past_date in data.index}
            past_returns = {name: ret for name, ret in past_returns.items() if not ret.empty}
            portfolio = proximal_grading_optimal(past_returns, initial_investment)
        elif sub_choice == '3':
            selected_stocks = request.form.getlist('selected_stocks')
            portfolio = proximal_grading_optimal_selected(selected_stocks, initial_investment)
        return render_template('optimize_portfolio.html', portfolio=portfolio, stocks=stocks.keys())
    return render_template('optimize_portfolio.html', stocks=stocks.keys())

@app.route('/worst_performing', methods=['GET', 'POST'])
def worst_performing_route():
    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        worst_performing_stocks = get_worst_performing_stocks(start_date, end_date)
        return render_template('worst_performing.html', worst_performing_stocks=worst_performing_stocks)
    return render_template('worst_performing.html')

@app.route('/visualize_performance')
def visualize_performance_route():
    plot_url = visualize_performance()
    return render_template('visualize_performance.html', plot_url=plot_url)

@app.route('/predict_returns', methods=['GET', 'POST'])
def predict_returns_route():
    mean_return = None
    cov_return = None
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        mean_return, cov_return = predict_stock_returns(stock_name)
    return render_template('predict_returns.html', mean_return=mean_return, cov_return=cov_return)


@app.route('/visualize_stock', methods=['GET', 'POST'])
def visualize_stock_route():
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        plot_url = visualize_stock_performance(stock_name, start_date, end_date)
        return render_template('visualize_stock.html', plot_url=plot_url)
    return render_template('visualize_stock.html')

@app.route('/visualize_selected', methods=['GET', 'POST'])
def visualize_selected_route():
    if request.method == 'POST':
        selected_stocks = request.form.getlist('selected_stocks')
        plot_url = visualize_selected_stocks_performance(selected_stocks)
        return render_template('visualize_selected.html', plot_url=plot_url, stocks=stocks.keys())
    return render_template('visualize_selected.html', stocks=stocks.keys())

if __name__ == '__main__':
    app.run(debug=True)
