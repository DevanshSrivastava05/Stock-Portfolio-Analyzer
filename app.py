import os
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ta import trend, volatility, momentum
from plotly.subplots import make_subplots
import numpy as np
import requests
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
import time
from functools import lru_cache

# DISCLAIMER: MUST BE PLACED AT TOP OF THE SCRIPT
DISCLAIMER = """
⚠️ IMPORTANT DISCLAIMER ⚠️

THIS IS A PERSONAL PRACTICE PROJECT FOR EDUCATIONAL PURPOSES ONLY.

- NOT A SEBI REGISTERED INVESTMENT ADVISORY SERVICE
- NOT PROVIDING FINANCIAL ADVICE OR RECOMMENDATIONS
- DO NOT USE FOR ACTUAL TRADING DECISIONS
- RESULTS ARE HYPOTHETICAL AND SIMULATED
- PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS

USE AT YOUR OWN RISK. CONSULT A CERTIFIED FINANCIAL ADVISOR 
FOR PROFESSIONAL INVESTMENT GUIDANCE.
"""

# CONFIGURATION: API KEY PLACEMENT
NEWS_API_KEY = "a3189ea541784a6dbd732d156cf305dd"  # Replace with your actual NewsAPI key

# Setup logging
logging.basicConfig(filename='stock_portfolio.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

class AdvancedStockPortfolioAnalyzer:
    def __init__(self):
        # Initialize session state and logging
        if "portfolio_data" not in st.session_state:
            st.session_state.portfolio_data = []
        
        # Cache for performance optimization
        self.stock_data_cache = {}
        
        # Machine Learning Model Path
        self.ml_model_path = 'stock_prediction_model.pkl'

    def fetch_news_sentiment(self, symbol):
        """Fetch and analyze news sentiment for a stock"""
        try:
            # Remove .NS from symbol if present
            clean_symbol = symbol.replace('.NS', '')
            
            # IMPORTANT: Replace with your actual NewsAPI key
            url = f"https://newsapi.org/v2/everything?q={clean_symbol}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            news_data = response.json()
            
            # Process news articles
            processed_news = []
            sentiments = []
            
            for article in news_data.get('articles', [])[:5]:  # Limit to 5 latest articles
                title = article.get('title', '')
                description = article.get('description', '')
                url = article.get('url', '')
                published_at = article.get('publishedAt', '')
                
                # Sentiment analysis
                blob = TextBlob(title + " " + description)
                sentiment_score = blob.sentiment.polarity
                sentiments.append(sentiment_score)
                
                # Classify sentiment
                if sentiment_score > 0:
                    sentiment_label = "Positive 📈"
                elif sentiment_score < 0:
                    sentiment_label = "Negative 📉"
                else:
                    sentiment_label = "Neutral 🔄"
                
                processed_news.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'published_at': published_at,
                    'sentiment_score': round(sentiment_score, 2),
                    'sentiment_label': sentiment_label
                })
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'news': processed_news,
                'avg_sentiment': round(avg_sentiment, 2)
            }
        except Exception as e:
            logging.error(f"Sentiment Analysis Error for {symbol}: {e}")
            return {
                'news': [],
                'avg_sentiment': 0
            }

    def input_stocks(self):
        """Interactive stock input interface"""
        st.sidebar.header("Add Stocks to Portfolio")
        
        # Input fields for stock details
        symbol = st.sidebar.text_input("Enter NSE Stock Symbol (e.g., TATAMOTORS.NS)")
        buy_price = st.sidebar.number_input("Buy Price", min_value=0.0, step=0.01)
        quantity = st.sidebar.number_input("Quantity", min_value=0, step=1)
        sector = st.sidebar.selectbox("Select Sector", [
            "Automobile", "Banking", "Energy", "Financial Services", 
            "Infrastructure", "Healthcare","Metals and Mining",
            "Oil & Gas",
            "IT", "Defense","FMCG","Manufacturing", "Paints", 
             "Others"
        ])

        # Add stock button
        if st.sidebar.button("Add Stock to Portfolio"):
            if symbol and buy_price > 0 and quantity > 0:
                # Check if stock already exists
                existing_stock = next((stock for stock in st.session_state.portfolio_data if stock["Symbol"] == symbol), None)
                
                if existing_stock:
                    st.sidebar.warning(f"Stock {symbol} already in portfolio. Choose 'Update' option.")
                else:
                    # Add new stock to portfolio
                    st.session_state.portfolio_data.append({
                        "Symbol": symbol.upper(), 
                        "Buy Price": buy_price, 
                        "Quantity": quantity, 
                        "Sector": sector
                    })
                    st.sidebar.success(f"Added {symbol} to portfolio!")
            else:
                st.sidebar.error("Please fill all fields correctly")

        # Analyze Portfolio button
        if st.sidebar.button("Analyze Portfolio"):
            if st.session_state.portfolio_data:
                self.display_dashboard()
            else:
                st.sidebar.error("Please add stocks to your portfolio first")

        # Display current portfolio
        if st.session_state.portfolio_data:
            st.sidebar.subheader("Current Portfolio")
            portfolio_df = pd.DataFrame(st.session_state.portfolio_data)
            st.sidebar.dataframe(portfolio_df)

    def fetch_market_data(self):
        for stock in st.session_state.portfolio_data:
            symbol = stock["Symbol"]
            try:
                stock_obj = yf.Ticker(symbol)
                history = stock_obj.history(period="6mo")
                if not history.empty:
                    current_price = history["Close"].iloc[-1]
                    stock["Current Price"] = current_price
                    stock["Return (%)"] = round(((current_price - stock["Buy Price"]) / stock["Buy Price"]) * 100, 2)
                    # Convert history to dict for better serialization
                    stock["History"] = {
                        'Open': history["Open"].tolist(),
                        'High': history["High"].tolist(),
                        'Low': history["Low"].tolist(),
                        'Close': history["Close"].tolist(),
                        'Volume': history["Volume"].tolist(),
                        'Dates': history.index.tolist()
                    }
                else:
                    stock["Current Price"] = None
                    stock["Return (%)"] = None
                    stock["History"] = None
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")

    def calculate_support_resistance(self, history_dict):
        """Calculate support and resistance levels using moving averages."""
        if not history_dict:
            return None, None
            
        history = pd.DataFrame({
            'Close': history_dict['Close']
        }, index=pd.DatetimeIndex(history_dict['Dates']))
        
        short_ma = trend.sma_indicator(history["Close"], window=20)
        long_ma = trend.sma_indicator(history["Close"], window=50)

        support = round(short_ma.min(), 2) if short_ma is not None else None
        resistance = round(long_ma.max(), 2) if long_ma is not None else None

        return support, resistance

    def plot_price_chart_with_indicators(self, symbol, history_dict):
        if not history_dict:
            return
            
        # Convert dict back to DataFrame
        history = pd.DataFrame({
            'Open': history_dict['Open'],
            'High': history_dict['High'],
            'Low': history_dict['Low'],
            'Close': history_dict['Close'],
            'Volume': history_dict['Volume']
        }, index=pd.DatetimeIndex(history_dict['Dates']))
        # Create subplots: price chart on the left, indicators on the right
        fig = make_subplots(
            rows=1, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            column_widths=[0.7, 0.3]
        )

        # Add Candlestick Chart and Close Price Line to the first subplot (Price Chart)
        fig.add_trace(go.Candlestick(
            x=history.index,
            open=history["Open"],
            high=history["High"],
            low=history["Low"],
            close=history["Close"],
            name="Candlestick",
            increasing_line_color='green', decreasing_line_color='red',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="blue", width=2),
            hoverinfo='x+y+name',
        ), row=1, col=1)

        # Add indicators to the second subplot (Indicators Chart)
        ema_50 = trend.ema_indicator(history["Close"], window=50)
        ema_200 = trend.ema_indicator(history["Close"], window=200)
        bollinger_middle = trend.sma_indicator(history["Close"], window=20)
        bollinger_upper = bollinger_middle + (history["Close"].rolling(window=20).std() * 2)
        bollinger_lower = bollinger_middle - (history["Close"].rolling(window=20).std() * 2)

        fig.add_trace(go.Scatter(
            x=history.index,
            y=ema_50,
            mode="lines",
            name="EMA (50)",
            line=dict(color="green", width=2, dash='solid'),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=history.index,
            y=ema_200,
            mode="lines",
            name="EMA (200)",
            line=dict(color="red", width=2, dash='solid'),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=history.index,
            y=bollinger_upper,
            mode="lines",
            name="Bollinger Upper",
            line=dict(color="orange", width=2, dash='dot'),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=history.index,
            y=bollinger_middle,
            mode="lines",
            name="Bollinger Middle (SMA)",
            line=dict(color="yellow", width=2, dash='dot'),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=history.index,
            y=bollinger_lower,
            mode="lines",
            name="Bollinger Lower",
            line=dict(color="orange", width=2, dash='dot'),
        ), row=1, col=2)

        # Update layout with the same dragmode for both price and indicator chart
        fig.update_layout(
            title=f"",  # Title in bold
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=True),  # Allow zooming on the price chart using the range slider
                type="date",
                tickformat="%b %d, %Y",
                showgrid=True,
                tickangle=-45,
                fixedrange=False,  # Disable zooming on the x-axis directly (for both)
                showticklabels=True,
            ),
            yaxis=dict(
                title="Price",
                fixedrange=False,  # Allow vertical movement (dragging)
            ),
            xaxis2=dict(
                title="Indicators",
                fixedrange=False,  # Allow panning in the indicator chart
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#f5f5f5",
            margin=dict(t=50, b=50, l=60, r=20),
            title_font=dict(size=16, color="black", family="Arial, sans-serif"),
            legend=dict(
                title="Indicators",
                x=0.01,
                y=1.05,
                orientation="h",
                bgcolor="rgba(255, 255, 255, 0.5)",
                font=dict(size=12)
            ),
            template="plotly_dark",
            hovermode="x unified",
            dragmode="pan"  # Enable pan mode for dragging on both price and indicator chart
        )

        # Adjust the annotation placement at the bottom right of the price chart
        fig.add_annotation(
            x=1.0,
            y=-0.5,
            xref="paper",
            yref="paper",
            text=f"<b>{symbol}</b>",
            showarrow=False,
            font=dict(size=17, color="BLACK"),
            align="right",
            xanchor="right",
            yanchor="bottom"
        )

        # Render the chart
        st.plotly_chart(fig, use_container_width=True, key=f"{symbol}_advanced_chart")

    def display_support_and_buy_recommendations(self):
        recommendations = []
        for stock in st.session_state.portfolio_data:
            if stock["History"] is not None:
                history = stock["History"]
                support, resistance = self.calculate_support_resistance(history)

                recommendations.append({
                    "Symbol": stock["Symbol"],
                    "Current Price": stock["Current Price"],
                    "Buy Price": stock["Buy Price"],
                    "Support Level": support,
                    "Resistance Level": resistance,
                    "Recommendation": "Buy" if stock["Current Price"] <= support else "Hold"
                })

        st.write("### Support Levels and Buy Recommendations")
        recommendations_df = pd.DataFrame(recommendations)
        st.dataframe(recommendations_df)

    def plot_sector_analysis(self):
        """Plot the portfolio's sector-wise allocation."""
        sector_df = pd.DataFrame(st.session_state.portfolio_data)
        sector_df["Total Buy Value"] = sector_df["Buy Price"] * sector_df["Quantity"]
        sector_summary = sector_df.groupby("Sector")["Total Buy Value"].sum()
        total_investment = sector_summary.sum()

        sector_percentage = (sector_summary / total_investment) * 100

        fig = go.Figure(data=[
            go.Bar(
                x=sector_summary.index,
                y=sector_summary.values,
                text=[f"{val:.2f}%" for val in sector_percentage.values],
                textposition="auto"
            )
        ])
        fig.update_layout(
            title="Sector-Wise Portfolio Allocation",
            xaxis_title="Sector",
            yaxis_title="Total Buy Value (₹)"
        )
        st.plotly_chart(fig)

    def display_news_sentiment(self):
        """Display news sentiment for each stock in portfolio"""
        st.write("### News Sentiment Analysis")
        
        for stock in st.session_state.portfolio_data:
            symbol = stock["Symbol"]
            
            # Fetch news sentiment
            news_data = self.fetch_news_sentiment(symbol)
            
            # Create expandable section for each stock
            with st.expander(f"News Sentiment for {symbol}"):
                # Display average sentiment
                st.metric("Average Sentiment Score", 
                          news_data['avg_sentiment'], 
                          help="Sentiment ranges from -1 (very negative) to 1 (very positive)")
                
                # Display news articles
                if news_data['news']:
                    for idx, article in enumerate(news_data['news'], 1):
                        st.markdown(f"""
                        **Article {idx}**
                        - **Title**: {article['title']}
                        - **Sentiment**: {article['sentiment_label']} (Score: {article['sentiment_score']})
                        - **Published At**: {article['published_at']}
                        - [Read Full Article]({article['url']})
                        ---
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent news found for this stock.")
    

    def display_dashboard(self):
        self.fetch_market_data()
        st.title("Advanced Stock Portfolio Analyzer")
        st.write("### Portfolio Summary")
        portfolio_df = pd.DataFrame(st.session_state.portfolio_data)
        st.dataframe(portfolio_df[["Symbol", "Buy Price", "Current Price", "Return (%)"]])

        st.write("### Sector Allocation")
        self.plot_sector_analysis()

        st.write("### Support and Buy Recommendations")
        self.display_support_and_buy_recommendations()

        # Add News Sentiment Analysis
        self.display_news_sentiment()

       


        for stock in st.session_state.portfolio_data:
            if stock["History"] is not None:
                self.plot_price_chart_with_indicators(stock["Symbol"], stock["History"])
def main():
    st.set_page_config(page_title="Advanced Stock Portfolio Analyzer", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dataframe {
        font-size: 12px;
    }
    .stMetric {
        background-color: #f1f3f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display Disclaimer
    st.markdown(f"""
    # 🚨 {DISCLAIMER}
    """, unsafe_allow_html=True)
    
    # Create an instance of the analyzer
    analyzer = AdvancedStockPortfolioAnalyzer()
   
    # Introduction text with enhanced formatting
    st.markdown("""
    ## 📊 WELCOME TO THE STOCK PORTFOLIO ANALYZER

    ### How to Use:
    1. 🔍 Use the sidebar to add stocks to your portfolio
    2. 📝 Specify details:
       - NSE Stock Symbol (e.g., TATAMOTORS.NS)
       - Buy Price
       - Quantity
       - Sector
    3. 🚀 Click "Add Stock to Portfolio" for each stock
    4. 📈 Click "Analyze Portfolio" when done
    """)
    
    # Portfolio input section
    analyzer.input_stocks()

    # Footer with additional information
    st.markdown("""
    ---
    ### 💡 Investment Disclaimer
    - This tool is for educational purposes
    - Always consult a financial advisor
    - Do not make investment decisions solely based on this analysis
    """)

if __name__ == "__main__":
    main()
