import yfinance as yf
import pandas as pd
import time
import requests

def get_nse_bse_tickers():
    print("Fetching NSE and BSE tickers...")
    
    nse_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    bse_url = "https://www.bseindia.com/stock-share-price/stockreach_v1/StockReachDownload.aspx?type=EQ"
    
    nse_tickers = []
    bse_tickers = []
    
    try:
        # NSE
        nse_df = pd.read_csv(nse_url)
        nse_tickers = nse_df['SYMBOL'].tolist()
        print(f"Fetched {len(nse_tickers)} NSE tickers")
    except Exception as e:
        print(f"Error fetching NSE tickers: {str(e)}")
    
    try:
        # BSE
        bse_response = requests.get(bse_url)
        bse_df = pd.read_csv(pd.compat.StringIO(bse_response.text))
        bse_tickers = bse_df['Security Id'].tolist()
        print(f"Fetched {len(bse_tickers)} BSE tickers")
    except Exception as e:
        print(f"Error fetching BSE tickers: {str(e)}")
    
    return nse_tickers, bse_tickers

def get_all_tickers():
    print("Fetching list of all tickers...")
    
    all_tickers = []
    
    # List of common stock exchanges
    exchanges = [
        "us_market", "nasdaq", "nyse", "amex", "toronto", "frankfurt", "paris", 
        "london", "amsterdam", "hong_kong", "tokyo", "sydney"
    ]
    
    for exchange in exchanges:
        try:
            tickers = yf.Ticker(f"^{exchange.upper()}").info.get('components', [])
            all_tickers.extend(tickers)
            print(f"Fetched tickers from {exchange}")
            time.sleep(1)  # To avoid overwhelming the API
        except Exception as e:
            print(f"Error fetching tickers from {exchange}: {str(e)}")
    
    # Add NSE and BSE tickers
    nse_tickers, bse_tickers = get_nse_bse_tickers()
    all_tickers.extend([f"{ticker}.NS" for ticker in nse_tickers])
    all_tickers.extend([f"{ticker}.BO" for ticker in bse_tickers])
    
    # Remove duplicates and sort
    all_tickers = sorted(list(set(all_tickers)))
    
    return all_tickers

def download_ticker_info(tickers):
    print("Downloading ticker information...")
    
    ticker_info = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            ticker_info.append({
                'Symbol': ticker,
                'Name': info.get('longName', ''),
                'Exchange': info.get('exchange', ''),
                'Sector': info.get('sector', ''),
                'Industry': info.get('industry', '')
            })
            print(f"Downloaded info for {ticker}")
            time.sleep(0.5)  # To avoid overwhelming the API
        except Exception as e:
            print(f"Error downloading info for {ticker}: {str(e)}")
    
    return ticker_info

def save_to_excel(data):
    df = pd.DataFrame(data)
    excel_filename = 'global_stock_tickers_with_india.xlsx'
    df.to_excel(excel_filename, index=False)
    print(f"Saved {len(data)} tickers to {excel_filename}")

if __name__ == "__main__":
    all_tickers = get_all_tickers()
    ticker_info = download_ticker_info(all_tickers)
    save_to_excel(ticker_info)