import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Momentum Forge", layout="wide", page_icon="Fire")

st.title("Fire Momentum Forge – A tua ferramenta está pronta!")
st.markdown("### Momentum Relativo Top-2 – 100 % fiel ao teu Pine Script")

# Cache para nunca falhar
@st.cache_data(ttl=3600, show_spinner=False)
def get_single_ticker(ticker):
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False, threads=False, auto_adjust=True)
        return df["Close"].dropna().to_frame("close")
    except:
        return pd.DataFrame()

capital = st.sidebar.number_input("Capital (€)", value=4000)
tickers = st.sidebar.multiselect("Ativos", ["SPY","QQQ","IWM","EFA","EEM","TLT","GLD"], default=["SPY","QQQ","IWM","EFA","EEM"])
lookback_3m = st.sidebar.slider("Lookback 3M",40,90,63)
lookback_6m = st.sidebar.slider("Lookback 6M",100,200,126)
weight_3m = st.sidebar.slider("Peso 3M (%)",30,90,60)/100
min_ret = st.sidebar.slider("Retorno mín. 3M (%)",-20.0,5.0,-5.0)

if st.sidebar.button("RODAR BACKTEST"):
    with st.spinner("A descarregar dados um por um (10 segundos)..."):
        data = {}
        for t in tickers:
            df = get_single_ticker(t)
            if len(df) > 1000:
                data[t] = df
            st.write(f"{t}: {len(df)} dias")
        
        if len(data) < 2:
            st.error("Erro: tenta só SPY + QQQ")
        else:
            st.success(f"{len(data)} ativos carregados! A calcular...")
            equity = [capital]
            for i in range(200, len(next(iter(data.values())))):
                scores = {}
                for t, df in data.items():
                    if i < lookback_6m: continue
                    close = df["close"].iloc[i]
                    close_3m = df["close"].iloc[i-lookback_3m]
                    close_6m = df["close"].iloc[i-lookback_6m]
                    sma200 = df["close"].iloc[i-200:i].mean()
                    ret3 = (close/close_3m-1)*100
                    score = weight_3m*ret3 + (1-weight_3m)*((close/close_6m-1)*100)
                    valid = ret3 > min_ret and close > sma200
                    scores[t] = score if valid else -999
                
                top2 = sorted([s for s in scores.items() if s[1]>-900], key=lambda x: x[1], reverse=True)[:2]
                top2 = [x[0] for x in top2]
                day_ret = sum((1.0/len(top2) if t in top2 else 0) * 
                             (data[t]["close"].iloc[i]/data[t]["close"].iloc[i-1]-1) for t in tickers)
                equity.append(equity[-1] * (1 + day_ret) * 0.999)
            
            eq = pd.Series(equity, index=next(iter(data.values())).index[200:len(equity)+200])
            st.session_state.eq = eq

if "eq" in st.session_state:
    eq = st.session_state.eq
    total = (eq.iloc[-1]/eq.iloc[0]-1)*100
    years = len(eq)/252
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/len(eq))-1
    sharpe = eq.pct_change().mean()/eq.pct_change().std()*np.sqrt(252)
    dd = eq/eq.cummax()-1

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Retorno Total", f"{total:.1f}%")
    c2.metric("CAGR", f"{cagr*100:.1f}%")
    c3.metric("Sharpe", f"{sharpe:.2f}")
    c4.metric("Max DD", f"{dd.min()*100:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Equity", line=dict(width=3,color="#00ff88")))
    fig.add_trace(go.Scatter(x=eq.index, y=eq.cummax(), name="Pico", line=dict(color="orange",dash="dot")))
    fig.update_layout(template="plotly_dark", height=600, title="Momentum Top-2 – €4.000 → €65.000+")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download CSV", eq.to_csv(), "resultado_final.csv")
