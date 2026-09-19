import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import json
from google import genai
from pydantic import BaseModel

# 1. Setup the Web Page
st.set_page_config(page_title="BTC Pattern Oracle", layout="wide")
st.title("🔮 BTC 15-Minute Pattern Oracle")
st.markdown("Pulls live internet data, calculates mathematical trends, and guesses the next move.")

# 2. Define what the AI needs to return
class Prediction(BaseModel):
    direction: str
    confidence: int
    reasoning: str

# 3. The Math Engine (Fetches data and calculates patterns)
@st.cache_data(ttl=30)
def fetch_and_calculate():
    # Using Kraken as it has reliable, free public data for BTC/USD
    exchange = ccxt.kraken()
    bars = exchange.fetch_ohlcv('BTC/USD', timeframe='1m', limit=60)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate Mathematical Trends
    df['EMA_9'] = ta.ema(df['close'], length=9)
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    
    # Drop rows with empty math (the first few minutes before averages can calculate)
    return df.dropna()

# 4. The Dashboard User Interface
api_key = st.text_input("Enter your Gemini API Key:", type="password")

if st.button("Scan Market & Predict"):
    if not api_key:
        st.warning("Please enter your API key first!")
    else:
        try:
            with st.spinner("Fetching live data and calculating math..."):
                df = fetch_and_calculate()
                
                # Draw an interactive chart
                fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                            open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name="Price")])
                
                # Overlay the math
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_9'], line=dict(color='orange', width=1.5), name="9-Min EMA"))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_21'], line=dict(color='blue', width=1.5), name="21-Min EMA"))
                
                fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show current stats
                latest = df.iloc[-1]
                st.write(f"**Current Price:** ${latest['close']:,.2f} | **RSI Momentum:** {latest['RSI_14']:.2f}")
                
                st.info("🧠 AI is analyzing the mathematical patterns...")
                
                # Initialize Gemini client
                client = genai.Client(api_key=api_key)
                
                # Send the last 15 minutes of math to the AI
                recent_data = df.tail(15).to_string()
                prompt = f"""
                You are a quantitative pattern recognition AI. Analyze this 15-minute BTC data including Moving Average trends and RSI momentum:
                
                {recent_data}
                
                Based strictly on these mathematical patterns, will the price be Higher ('UP') or Lower ('DOWN') 15 minutes from now?
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={'response_mime_type': 'application/json', 'response_schema': Prediction}
                )
                
                result = json.loads(response.text)
                
                # Print the final verdict
                st.success(f"### 📈 PREDICTION: {result['direction']} ({result['confidence']}% Confidence)")
                st.write(f"**Pattern Logic:** {result['reasoning']}")
                
        except Exception as e:
            st.error(f"An error occurred. Make sure your API key is valid. Details: {e}")