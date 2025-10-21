import streamlit as st
from exploration import black_scholes, delta, solve_for_implied_volatility
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

st.title("Options Pricing Learning")


S = 262.8
K = 170
T = 0.02
r = 0.05
sigma = 0.2
q = 0.0

# call = black_scholes(S, K, T, r, sigma, q, "call")
# put = black_scholes(S, K, T, r, sigma, q, "put")

# print(call)
# print(put)

# print(delta(S, K, T, r, sigma, q, "call"))

# iv = solve_for_implied_volatility(S, K, T, r, 94.18, q, sigma_guess=1.8, option_type="call")
# print(f"Implied Volatility for Call Option: {iv}")



df_calls = pd.read_csv("../data/aapl_call_option_chain.csv")
df_calls["Option Type"] = "call"
df_calls["Maturity Date"] = pd.to_datetime(df_calls["Maturity Date"])
today = pd.to_datetime("today").normalize()
df_calls["T"] = (df_calls["Maturity Date"] - today).dt.days / 365.0

df_iv_curve = pd.DataFrame()

S = 262.24 # Current stock price of AAPL

def compute_iv(row):
    K = row["Strike"]
    T = row["T"]
    market_price = row["Last Price"]

    return solve_for_implied_volatility(S, K, T, r, market_price, q=0, sigma_guess=0.5, option_type="call")

df_iv_curve["Moneyness"] = S / df_calls["Strike"]
df_iv_curve["Time to Maturity"] = df_calls["T"]
df_iv_curve["Calculated IV"] = df_calls.apply(compute_iv, axis=1)
df_iv_curve["Suggested IV"] = df_calls["Implied Volatility"].apply(lambda x: float(x.strip('%')))

print(df_iv_curve)


def plot_iv_surface(df, nx=60, ny=60):
    # require enough valid points
    df = df.dropna(subset=['Moneyness', 'Time to Maturity', 'Calculated IV']).copy()
    if len(df) < 5:
        st.warning("Not enough data points to build a surface — showing scatter instead.")
        fig = go.Figure(data=go.Scatter3d(
            x=df['Moneyness'], y=df['Time to Maturity'], z=df['Calculated IV'],
            mode='markers', marker=dict(size=4)
        ))
        st.plotly_chart(fig, use_container_width=True)
        return

    pts = df[['Moneyness', 'Time to Maturity']].values
    vals = df['Calculated IV'].values

    # build a regular grid over the extents
    xi = np.linspace(df['Moneyness'].min(), df['Moneyness'].max(), nx)
    yi = np.linspace(df['Time to Maturity'].min(), df['Time to Maturity'].max(), ny)
    XI, YI = np.meshgrid(xi, yi)

    # linear interpolation; where linear produces NaN, fill with nearest
    ZI = griddata(pts, vals, (XI, YI), method='linear')
    if np.isnan(ZI).any():
        ZI_nearest = griddata(pts, vals, (XI, YI), method='nearest')
        ZI = np.where(np.isnan(ZI), ZI_nearest, ZI)

    # final safety: if still NaNs, replace with small constant or drop
    nan_mask = np.isnan(ZI)
    if nan_mask.any():
        ZI[nan_mask] = np.nanmean(vals) if np.isfinite(np.nanmean(vals)) else 0.0

    fig = go.Figure(data=[go.Surface(x=XI, y=YI, z=ZI, colorscale='Viridis', showscale=True)])
    fig.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Moneyness (S/K)',
            yaxis_title='Time to Maturity (yrs)',
            zaxis_title='Implied Volatility',
            aspectmode='auto'  # or 'cube' / dict(x=...,y=...,z=...) to adjust scaling
        ),
        width=900,
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

# Add some controls
st.sidebar.header("Plot Controls")
plot_type = st.sidebar.selectbox(
    "Select Plot Type",
    ["3D Surface", "2D Scatter"]
)

if plot_type == "3D Surface":
    plot_iv_surface(df_iv_curve)
else:
    # 2D scatter plot as alternative view
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_iv_curve['Moneyness'],
        y=df_iv_curve['Suggested IV'],
        mode='markers',
        name='Calculated IV'
    ))
    
    fig.update_layout(
        title='Implied Volatility vs Moneyness',
        xaxis_title='Moneyness (S/K)',
        yaxis_title='Implied Volatility',
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Display the raw data if needed
if st.checkbox("Show raw data"):
    st.write(df_iv_curve)