"""
Momentum Forge - Plataforma Institucional de Backtesting
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    PlatformConfig,
    BacktestConfig,
    MomentumStrategyConfig,
    RiskConfig,
    RebalanceFrequency
)
from src.strategies import (
    MomentumStrategy,
    RiskAdjustedMomentumStrategy,
    DualMomentumStrategy,
    StrategyFactory,
    StrategyType,
    get_strategy_presets
)
from src.data import (
    YahooDataProvider,
    YahooDataProviderCached,
    DataFrequency,
    validate_backtest_data,
    DataCleaner,
    UNIVERSES
)
from src.core import (
    Backtester,
    BacktestResult,
    Portfolio,
    RiskManager
)
from src.analytics import (
    MetricsCalculator,
    generate_tearsheet,
    ReportGenerator
)


# ============================================================================
# CONFIGURACAO DA PAGINA
# ============================================================================

st.set_page_config(
    page_title="Momentum Forge | Institutional Backtesting",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .risk-alert {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #ff1744;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCOES AUXILIARES
# ============================================================================

@st.cache_data(ttl=3600)
def load_data_cached(symbols: tuple, start_date: str, end_date: str):
    """Carrega dados com cache"""
    provider = YahooDataProviderCached()

    data = provider.get_ohlcv_multiple(
        symbols=list(symbols),
        start_date=start_date,
        end_date=end_date,
        frequency=DataFrequency.DAILY
    )

    return data


def format_currency(value: float, currency: str = "USD") -> str:
    """Formata valor como moeda"""
    symbols = {"USD": "$", "EUR": "€", "BRL": "R$"}
    symbol = symbols.get(currency, "$")
    return f"{symbol}{value:,.2f}"


def format_pct(value: float) -> str:
    """Formata valor como percentual"""
    color = "positive" if value >= 0 else "negative"
    return f'<span class="{color}">{value:.2%}</span>'


def create_equity_chart(equity: pd.Series, benchmark: pd.Series = None) -> go.Figure:
    """Cria grafico de equity curve"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown")
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Portfolio",
            line=dict(color="#1f77b4", width=2)
        ),
        row=1, col=1
    )

    # Peak
    peak = equity.expanding().max()
    fig.add_trace(
        go.Scatter(
            x=peak.index,
            y=peak.values,
            name="Peak",
            line=dict(color="orange", dash="dot", width=1)
        ),
        row=1, col=1
    )

    # Benchmark
    if benchmark is not None and len(benchmark) > 0:
        # Normaliza para comecar no mesmo valor
        bench_normalized = benchmark / benchmark.iloc[0] * equity.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=bench_normalized.values,
                name="Benchmark",
                line=dict(color="gray", width=1, dash="dash")
            ),
            row=1, col=1
        )

    # Drawdown
    drawdown = (equity - peak) / peak
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#ff1744", width=1)
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    fig.update_yaxes(title_text="Valor", row=1, col=1)
    fig.update_yaxes(title_text="DD %", tickformat=".1%", row=2, col=1)

    return fig


def create_monthly_heatmap(returns: pd.Series) -> go.Figure:
    """Cria heatmap de retornos mensais"""
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    df = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Return': monthly.values
    })

    pivot = df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                     'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'][:len(pivot.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot.values * 100, 1),
        texttemplate="%{text:.1f}%",
        textfont={"size": 10},
        hovertemplate="Ano: %{y}<br>Mes: %{x}<br>Retorno: %{z:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title="Retornos Mensais (%)",
        height=400,
        template="plotly_dark"
    )

    return fig


def create_distribution_chart(returns: pd.Series) -> go.Figure:
    """Cria grafico de distribuicao de retornos"""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name="Retornos",
        marker_color="#1f77b4",
        opacity=0.7
    ))

    # Linha vertical na media
    mean_ret = returns.mean() * 100
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="green",
                  annotation_text=f"Media: {mean_ret:.2f}%")

    # Linha vertical no zero
    fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=1)

    fig.update_layout(
        title="Distribuicao de Retornos Diarios",
        xaxis_title="Retorno (%)",
        yaxis_title="Frequencia",
        height=350,
        template="plotly_dark"
    )

    return fig


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("## 📊 Momentum Forge")
st.sidebar.markdown("*Institutional Backtesting Platform*")
st.sidebar.markdown("---")

# Tabs na sidebar
tab_config, tab_strategy, tab_risk = st.sidebar.tabs(["Config", "Estrategia", "Risco"])

with tab_config:
    st.markdown("### Configuracao Geral")

    # Capital
    initial_capital = st.number_input(
        "Capital Inicial ($)",
        min_value=1000,
        max_value=100000000,
        value=100000,
        step=10000
    )

    # Periodo
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Data Inicio",
            value=datetime(2015, 1, 1),
            min_value=datetime(2000, 1, 1)
        )
    with col2:
        end_date = st.date_input(
            "Data Fim",
            value=datetime.now() - timedelta(days=1)
        )

    # Rebalanceamento
    rebalance_freq = st.selectbox(
        "Rebalanceamento",
        options=["Diario", "Semanal", "Mensal"],
        index=0
    )

    # Benchmark
    benchmark_symbol = st.text_input("Benchmark", value="SPY")

with tab_strategy:
    st.markdown("### Parametros da Estrategia")

    # Preset
    presets = get_strategy_presets()
    preset_name = st.selectbox(
        "Preset",
        options=["Custom"] + list(presets.keys()),
        index=0
    )

    if preset_name != "Custom":
        preset = presets[preset_name]
        lookback_short = preset['parameters'].get('lookback_short', 63)
        lookback_long = preset['parameters'].get('lookback_long', 126)
        weight_short = preset['parameters'].get('weight_short', 0.6)
        min_return = preset['parameters'].get('min_return', -0.05)
        top_n = preset['parameters'].get('top_n', 2)
    else:
        lookback_short = 63
        lookback_long = 126
        weight_short = 0.6
        min_return = -0.05
        top_n = 2

    # Parametros ajustaveis
    lookback_short = st.slider(
        "Lookback Curto (dias)",
        min_value=20,
        max_value=126,
        value=lookback_short
    )

    lookback_long = st.slider(
        "Lookback Longo (dias)",
        min_value=63,
        max_value=252,
        value=lookback_long
    )

    weight_short = st.slider(
        "Peso Momentum Curto",
        min_value=0.0,
        max_value=1.0,
        value=weight_short,
        step=0.1
    )

    min_return = st.slider(
        "Retorno Minimo",
        min_value=-0.20,
        max_value=0.10,
        value=min_return,
        step=0.01,
        format="%.0f%%"
    )

    top_n = st.slider(
        "Top N Ativos",
        min_value=1,
        max_value=10,
        value=top_n
    )

    # Universo
    universe_option = st.selectbox(
        "Universo de Ativos",
        options=list(UNIVERSES.keys()),
        index=0
    )

with tab_risk:
    st.markdown("### Limites de Risco")

    max_position = st.slider(
        "Max Posicao (%)",
        min_value=5,
        max_value=50,
        value=20
    ) / 100

    max_drawdown = st.slider(
        "Max Drawdown (%)",
        min_value=5,
        max_value=50,
        value=20
    ) / 100

    max_leverage = st.slider(
        "Max Alavancagem",
        min_value=1.0,
        max_value=3.0,
        value=1.0,
        step=0.1
    )

    # Custos
    st.markdown("### Custos de Transacao")

    commission_bps = st.number_input(
        "Comissao (bps)",
        min_value=0.0,
        max_value=50.0,
        value=5.0
    )

    slippage_bps = st.number_input(
        "Slippage (bps)",
        min_value=0.0,
        max_value=50.0,
        value=2.0
    )


# Botao de executar
st.sidebar.markdown("---")
run_backtest = st.sidebar.button(
    "🚀 EXECUTAR BACKTEST",
    use_container_width=True,
    type="primary"
)


# ============================================================================
# AREA PRINCIPAL
# ============================================================================

st.markdown('<h1 class="main-header">Momentum Forge</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Institutional Momentum Backtesting Platform</p>', unsafe_allow_html=True)
st.markdown("---")


if run_backtest:

    with st.spinner("Carregando dados..."):

        # Configura universo
        universe = UNIVERSES.get(universe_option, UNIVERSES['etf_diversified'])

        # Carrega dados
        try:
            data = load_data_cached(
                symbols=tuple(universe),
                start_date=str(start_date),
                end_date=str(end_date)
            )

            if data.empty:
                st.error("Erro ao carregar dados. Verifique sua conexao.")
                st.stop()

        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            st.stop()

        # Valida dados
        validation = validate_backtest_data(data, min_history=lookback_long + 50)

        if not validation.is_valid:
            st.warning(f"Aviso de qualidade de dados: {len(validation.issues)} problemas detectados")
            with st.expander("Ver detalhes"):
                for issue in validation.issues[:5]:
                    st.write(f"- {issue.message}")

    with st.spinner("Executando backtest..."):

        # Configura estrategia
        strategy_config = MomentumStrategyConfig(
            lookback_short=lookback_short,
            lookback_long=lookback_long,
            weight_short=weight_short,
            min_return=min_return,
            top_n=top_n,
            universe=universe
        )

        strategy = MomentumStrategy(config=strategy_config)

        # Configura risco
        risk_config = RiskConfig(
            max_position_size=max_position,
            max_drawdown_limit=max_drawdown,
            max_portfolio_leverage=max_leverage
        )

        # Configura backtest
        freq_map = {
            "Diario": RebalanceFrequency.DAILY,
            "Semanal": RebalanceFrequency.WEEKLY,
            "Mensal": RebalanceFrequency.MONTHLY
        }

        backtest_config = BacktestConfig(
            initial_capital=initial_capital,
            rebalance_frequency=freq_map[rebalance_freq],
            risk_config=risk_config,
            benchmark=benchmark_symbol,
            warmup_period=max(lookback_long, 200) + 10
        )
        backtest_config.transaction_costs.commission_bps = commission_bps
        backtest_config.transaction_costs.slippage_bps = slippage_bps

        # Cria e executa backtester
        provider = YahooDataProviderCached()
        backtester = Backtester(
            config=backtest_config,
            strategy=strategy,
            data_provider=provider
        )

        # Carrega benchmark
        try:
            benchmark_data = provider.get_ohlcv(
                symbol=benchmark_symbol,
                start_date=str(start_date),
                end_date=str(end_date)
            )
        except:
            benchmark_data = None

        # Executa
        result = backtester.run(data, benchmark_data)

        # Salva no estado
        st.session_state.result = result
        st.session_state.benchmark_data = benchmark_data
        st.session_state.config = {
            'capital': initial_capital,
            'start': start_date,
            'end': end_date,
            'strategy': strategy_config.__dict__
        }


# Exibe resultados
if 'result' in st.session_state:
    result = st.session_state.result
    config = st.session_state.config

    # ========================================================================
    # METRICAS PRINCIPAIS
    # ========================================================================

    st.markdown("### 📈 Performance Summary")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "Retorno Total",
            f"{result.total_return:.1%}",
            delta=f"{result.total_return - 0:.1%}"
        )

    with col2:
        st.metric(
            "CAGR",
            f"{result.cagr:.1%}"
        )

    with col3:
        st.metric(
            "Volatilidade",
            f"{result.annualized_volatility:.1%}"
        )

    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.2f}"
        )

    with col5:
        st.metric(
            "Max Drawdown",
            f"{result.max_drawdown:.1%}"
        )

    with col6:
        st.metric(
            "Calmar Ratio",
            f"{result.calmar_ratio:.2f}"
        )

    st.markdown("---")

    # ========================================================================
    # GRAFICO PRINCIPAL
    # ========================================================================

    # Prepara benchmark para grafico
    if 'benchmark_data' in st.session_state and st.session_state.benchmark_data is not None:
        bench_close = st.session_state.benchmark_data.get('Close',
                      st.session_state.benchmark_data.get('Adj Close'))
        if bench_close is not None:
            bench_returns = bench_close.pct_change().dropna()
            benchmark_equity = (1 + bench_returns).cumprod() * config['capital']
        else:
            benchmark_equity = None
    else:
        benchmark_equity = None

    # Grafico de equity
    equity_chart = create_equity_chart(result.equity_curve, benchmark_equity)
    st.plotly_chart(equity_chart, use_container_width=True)

    # ========================================================================
    # METRICAS DETALHADAS
    # ========================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metricas Detalhadas",
        "📅 Retornos Mensais",
        "📈 Distribuicao",
        "📋 Relatorio"
    ])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Retornos")
            st.write(f"**Total Return:** {result.total_return:.2%}")
            st.write(f"**CAGR:** {result.cagr:.2%}")
            st.write(f"**Volatilidade:** {result.annualized_volatility:.2%}")

        with col2:
            st.markdown("#### Risk-Adjusted")
            st.write(f"**Sharpe Ratio:** {result.sharpe_ratio:.2f}")
            st.write(f"**Sortino Ratio:** {result.sortino_ratio:.2f}")
            st.write(f"**Calmar Ratio:** {result.calmar_ratio:.2f}")

        with col3:
            st.markdown("#### Drawdown")
            st.write(f"**Max Drawdown:** {result.max_drawdown:.2%}")
            st.write(f"**Max DD Duration:** {result.max_drawdown_duration} dias")
            st.write(f"**Avg Drawdown:** {result.avg_drawdown:.2%}")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Tail Risk")
            st.write(f"**VaR 95%:** {result.var_95:.2%}")
            st.write(f"**CVaR 95%:** {result.cvar_95:.2%}")
            st.write(f"**Skewness:** {result.skewness:.2f}")
            st.write(f"**Kurtosis:** {result.kurtosis:.2f}")

        with col2:
            st.markdown("#### Benchmark")
            st.write(f"**Alpha:** {result.alpha:.2%}")
            st.write(f"**Beta:** {result.beta:.2f}")
            st.write(f"**Info Ratio:** {result.information_ratio:.2f}")
            st.write(f"**Tracking Error:** {result.tracking_error:.2%}")

        with col3:
            st.markdown("#### Trading")
            st.write(f"**Num Trades:** {result.num_trades}")
            st.write(f"**Win Rate:** {result.win_rate:.1%}")
            st.write(f"**Profit Factor:** {result.profit_factor:.2f}")
            st.write(f"**Turnover:** {result.turnover:.1%}")

    with tab2:
        heatmap = create_monthly_heatmap(result.returns)
        st.plotly_chart(heatmap, use_container_width=True)

        # Tabela de retornos anuais
        yearly = result.returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        yearly_df = pd.DataFrame({
            'Ano': yearly.index.year,
            'Retorno': yearly.values
        })
        yearly_df['Retorno'] = yearly_df['Retorno'].apply(lambda x: f"{x:.1%}")
        st.dataframe(yearly_df, use_container_width=True, hide_index=True)

    with tab3:
        dist_chart = create_distribution_chart(result.returns)
        st.plotly_chart(dist_chart, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Estatisticas")
            st.write(f"**Media Diaria:** {result.returns.mean():.4%}")
            st.write(f"**Mediana:** {result.returns.median():.4%}")
            st.write(f"**Desvio Padrao:** {result.returns.std():.4%}")
            st.write(f"**Dias Positivos:** {(result.returns > 0).sum()}")
            st.write(f"**Dias Negativos:** {(result.returns < 0).sum()}")

        with col2:
            st.markdown("#### Extremos")
            st.write(f"**Melhor Dia:** {result.returns.max():.2%}")
            st.write(f"**Pior Dia:** {result.returns.min():.2%}")
            st.write(f"**Melhor Mes:** {result.returns.resample('ME').apply(lambda x: (1+x).prod()-1).max():.2%}")
            st.write(f"**Pior Mes:** {result.returns.resample('ME').apply(lambda x: (1+x).prod()-1).min():.2%}")

    with tab4:
        # Gera tearsheet
        tearsheet = generate_tearsheet(
            result.returns,
            title="Momentum Strategy Tearsheet"
        )

        # Exibe JSON formatado
        st.json(tearsheet['full_metrics'])

        # Download buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV de equity
            csv_equity = result.equity_curve.to_csv()
            st.download_button(
                "📥 Download Equity CSV",
                csv_equity,
                "equity_curve.csv",
                "text/csv"
            )

        with col2:
            # CSV de trades
            if not result.trades_history.empty:
                csv_trades = result.trades_history.to_csv()
                st.download_button(
                    "📥 Download Trades CSV",
                    csv_trades,
                    "trades.csv",
                    "text/csv"
                )

        with col3:
            # JSON de metricas
            import json
            json_metrics = json.dumps(result.to_dict(), indent=2, default=str)
            st.download_button(
                "📥 Download Metricas JSON",
                json_metrics,
                "metrics.json",
                "application/json"
            )

else:
    # Estado inicial
    st.info("👈 Configure os parametros na barra lateral e clique em **EXECUTAR BACKTEST**")

    st.markdown("""
    ### Bem-vindo ao Momentum Forge

    Esta plataforma institucional permite backtesting de estrategias de momentum com:

    - **Estrategias de Momentum**: Top-N, Risk-Adjusted, Dual Momentum
    - **Gestao de Risco**: Limites de posicao, drawdown, alavancagem
    - **Custos Realistas**: Comissao, slippage, market impact
    - **Metricas Completas**: Sharpe, Sortino, VaR, Alpha, Beta e mais
    - **Dados de Mercado**: Yahoo Finance (gratuito)

    #### Universos Disponiveis
    """)

    for name, symbols in UNIVERSES.items():
        st.write(f"**{name}:** {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "Momentum Forge v2.0 | Institutional Backtesting Platform | "
    f"Dados: Yahoo Finance | Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    "</div>",
    unsafe_allow_html=True
)
