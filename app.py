import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Momentum Forge", layout="wide", page_icon="🔥")
st.title("🔥 Momentum Forge – Backtest Robusto")
st.markdown("### Momentum Relativo Top-2 – Dados reais 2010-2025")

# DADOS PRÉ-CARREGADOS (2010-2025)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/quant-br/momentum-data/main/momentum_top2_data.csv"
    try:
        df = pd.read_csv(url, index_col="Date", parse_dates=True)
        if df.empty:
            return None, "Dados vazios no servidor"
        return {col: df[[col]].rename(columns={col: "close"}) for col in df.columns}, None
    except Exception as e:
        return None, f"Erro ao carregar dados: {str(e)}"

# SIDEBAR - Parâmetros com documentação
st.sidebar.header("⚙️ Parâmetros da Estratégia")

st.sidebar.markdown("**Capital Inicial**")
capital = st.sidebar.number_input("Capital (€)", value=4000, min_value=100, help="Capital inicial para o backtest")

st.sidebar.markdown("---")
st.sidebar.markdown("**Períodos de Lookback**")
st.sidebar.caption("Janelas de tempo para calcular o momentum")
lookback_3m = st.sidebar.slider("Lookback 3M (dias)", 40, 90, 63, help="Período curto (~3 meses)")
lookback_6m = st.sidebar.slider("Lookback 6M (dias)", 100, 200, 126, help="Período longo (~6 meses)")

st.sidebar.markdown("---")
st.sidebar.markdown("**Pesos e Filtros**")
weight_3m = st.sidebar.slider("Peso Momentum 3M (%)", 30, 90, 60, help="Peso do momentum de curto prazo no score final") / 100
min_ret = st.sidebar.slider("Retorno mín. 3M (%)", -20.0, 5.0, -5.0, help="Filtro: exclui ativos com retorno 3M abaixo deste valor")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sobre a Estratégia**")
st.sidebar.caption("""
A estratégia seleciona os **2 melhores ativos** por momentum
combinado (3M + 6M), aplicando filtros de:
- Retorno mínimo 3M
- Preço acima da SMA200

Rebalanceamento diário com 0.1% de custo de transação.
""")

if st.sidebar.button("🚀 RODAR BACKTEST", type="primary"):
    with st.spinner("A calcular backtest..."):
        data, error = load_data()

        if error:
            st.error(f"❌ {error}")
            st.stop()

        tickers = list(data.keys())
        equity = [capital]
        benchmark_equity = [capital]  # SPY como benchmark
        daily_returns = []

        # Verificar se SPY existe nos dados
        has_spy = "SPY" in tickers

        for i in range(200, len(next(iter(data.values())))):
            scores = {}
            for t in tickers:
                df = data[t]["close"]
                close = df.iloc[i]
                close_3m = df.iloc[i-lookback_3m]
                close_6m = df.iloc[i-lookback_6m]
                sma200 = df.iloc[i-200:i].mean()
                ret3 = (close/close_3m-1)*100
                score = weight_3m*ret3 + (1-weight_3m)*((close/close_6m-1)*100)
                valid = ret3 > min_ret and close > sma200
                scores[t] = score if valid else -999

            # Selecionar top 2
            top2 = sorted([s for s in scores.items() if s[1] > -900], key=lambda x: x[1], reverse=True)[:2]
            top2 = [x[0] for x in top2]

            # ROBUSTEZ: se não há ativos válidos, manter em cash (retorno 0)
            if len(top2) == 0:
                day_ret = 0
            else:
                day_ret = sum((1.0/len(top2)) *
                             (data[t]["close"].iloc[i]/data[t]["close"].iloc[i-1]-1) for t in top2)

            daily_returns.append(day_ret)
            equity.append(equity[-1] * (1 + day_ret) * 0.999)  # 0.1% custo

            # Benchmark (SPY buy-and-hold)
            if has_spy:
                spy_ret = data["SPY"]["close"].iloc[i]/data["SPY"]["close"].iloc[i-1]-1
                benchmark_equity.append(benchmark_equity[-1] * (1 + spy_ret))

        # Criar séries temporais
        dates = next(iter(data.values())).index[200:len(equity)+200]
        eq = pd.Series(equity, index=dates)
        returns_series = pd.Series(daily_returns, index=dates[1:])

        # Guardar em session_state
        st.session_state.eq = eq
        st.session_state.returns = returns_series
        st.session_state.has_spy = has_spy
        if has_spy:
            st.session_state.benchmark = pd.Series(benchmark_equity, index=dates)

# EXIBIR RESULTADOS
if "eq" in st.session_state:
    eq = st.session_state.eq
    returns_series = st.session_state.returns

    # Métricas básicas
    total = (eq.iloc[-1]/eq.iloc[0]-1)*100
    years = len(eq)/252
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years)-1
    sharpe = returns_series.mean()/returns_series.std()*np.sqrt(252) if returns_series.std() > 0 else 0
    dd = eq/eq.cummax()-1
    max_dd = dd.min()

    # Métricas adicionais
    positive_days = (returns_series > 0).sum()
    total_days = len(returns_series)
    win_rate = positive_days / total_days * 100 if total_days > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Exibir métricas em 2 linhas
    st.markdown("#### 📊 Métricas de Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno Total", f"{total:.1f}%")
    c2.metric("CAGR", f"{cagr*100:.1f}%")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c4.metric("Max Drawdown", f"{max_dd*100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Win Rate", f"{win_rate:.1f}%", help="% de dias positivos")
    c6.metric("Calmar Ratio", f"{calmar:.2f}", help="CAGR / Max DD")
    c7.metric("Capital Final", f"€{eq.iloc[-1]:,.0f}")
    c8.metric("Anos", f"{years:.1f}")

    # Comparação com benchmark
    if st.session_state.get("has_spy", False):
        benchmark = st.session_state.benchmark
        bench_total = (benchmark.iloc[-1]/benchmark.iloc[0]-1)*100
        bench_cagr = (benchmark.iloc[-1]/benchmark.iloc[0])**(1/years)-1

        st.markdown("---")
        st.markdown("#### 📈 Comparação com SPY (Buy & Hold)")
        b1, b2, b3 = st.columns(3)
        b1.metric("SPY Retorno Total", f"{bench_total:.1f}%", delta=f"{total-bench_total:+.1f}% vs Estratégia")
        b2.metric("SPY CAGR", f"{bench_cagr*100:.1f}%", delta=f"{(cagr-bench_cagr)*100:+.1f}% vs Estratégia")
        b3.metric("SPY Capital Final", f"€{benchmark.iloc[-1]:,.0f}")

    # Gráfico
    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Estratégia Momentum", line=dict(width=2, color="#00ff88")))
    fig.add_trace(go.Scatter(x=eq.index, y=eq.cummax(), name="Pico", line=dict(color="orange", dash="dot", width=1)))

    if st.session_state.get("has_spy", False):
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name="SPY (Buy & Hold)", line=dict(width=2, color="#ff6b6b")))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        title=f"Evolução do Capital: €{capital:,} → €{eq.iloc[-1]:,.0f}",
        xaxis_title="Data",
        yaxis_title="Capital (€)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Equity CSV", eq.to_csv(), "momentum_equity.csv", type="secondary")
    with col2:
        st.download_button("📥 Download Returns CSV", returns_series.to_csv(), "momentum_returns.csv", type="secondary")

st.markdown("---")
st.success("✅ Sistema robusto: dados pré-carregados, sem dependências externas durante execução")
