import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter

from implied_volatility import BrentIVCalculator
from pricing import BlackScholesModel

def render():
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


    df_calls = pd.read_csv("../data/live_calls.csv")
    df_calls["Option Type"] = "call"
    df_calls["Maturity Date"] = pd.to_datetime(df_calls["expiration"])
    today = pd.to_datetime("today").normalize()
    df_calls["T"] = (df_calls["Maturity Date"] - today).dt.days / 365.0

    df_iv_curve = pd.DataFrame()

    S = 262.24 # Current stock price of AAPL

    def compute_iv(row):
        K = row["strike"]
        T = row["T"]
        market_price = row["lastPrice"]

        iv_calculator = BrentIVCalculator(BlackScholesModel(), vol_min=0.001, vol_max=5.0)
        return iv_calculator.solve_for_iv(S, K, T, r, q, market_price, option_type="call")

    df_iv_curve["Moneyness"] = S / df_calls["strike"]
    df_iv_curve["Time to Maturity"] = df_calls["T"]
    df_iv_curve["Calculated IV"] = df_calls.apply(compute_iv, axis=1)
    df_iv_curve["Suggested IV"] = df_calls["impliedVolatility"]

    def plot_smoothed_iv_surface(df, nx=60, ny=60, smoothing_method='rbf', smoothness=1.0):
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
        
        # Remove extreme outliers
        q_low, q_high = np.percentile(vals, [5, 95])
        mask = (vals >= q_low) & (vals <= q_high)
        pts = pts[mask]
        vals = vals[mask]
        
        xi = np.linspace(df['Moneyness'].min(), df['Moneyness'].max(), nx)
        yi = np.linspace(df['Time to Maturity'].min(), df['Time to Maturity'].max(), ny)
        XI, YI = np.meshgrid(xi, yi)
        grid_pts = np.column_stack([XI.ravel(), YI.ravel()])

        if smoothing_method == 'rbf':
            # Radial Basis Function interpolation (smooth)
            interpolator = RBFInterpolator(pts, vals, kernel='thin_plate_spline', smoothing=smoothness)
            ZI = interpolator(grid_pts).reshape(XI.shape)
            
        # elif smoothing_method == 'ct':
        #     # Clough-Tocher (C1 smooth piecewise cubic)
        #     interpolator = CloughTocher2DInterpolator(pts, vals, fill_value=0.0)
        #     ZI = interpolator(XI, YI)
            
        elif smoothing_method == 'gaussian':
            # Linear interpolation + Gaussian smoothing
            ZI = griddata(pts, vals, (XI, YI), method='linear')
            if np.isnan(ZI).any():
                ZI_nearest = griddata(pts, vals, (XI, YI), method='nearest')
                ZI = np.where(np.isnan(ZI), ZI_nearest, ZI)
            # Apply Gaussian smoothing
            ZI = gaussian_filter(ZI, sigma=smoothness)
            
        else:  # original method
            ZI = griddata(pts, vals, (XI, YI), method='linear')
            if np.isnan(ZI).any():
                ZI_nearest = griddata(pts, vals, (XI, YI), method='nearest')
                ZI = np.where(np.isnan(ZI), ZI_nearest, ZI)

        # Ensure positive volatilities
        ZI = np.maximum(ZI, 0.01)
        
        fig = go.Figure(data=[go.Surface(x=XI, y=YI, z=ZI, colorscale='Viridis', showscale=True)])
        fig.update_layout(
            title=f'Smoothed Implied Volatility Surface ({smoothing_method})',
            scene=dict(
                xaxis_title='Moneyness (S/K)',
                yaxis_title='Time to Maturity (yrs)',
                zaxis_title='Implied Volatility',
                aspectmode='auto'
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

    # Add these controls to your sidebar
    st.sidebar.header("Volatility Surface Smoothing")
    smoothing_method = st.sidebar.selectbox(
        "Smoothing Method",
        ["rbf", "ct", "gaussian", "none"],
        index=0,
        help="RBF: Radial Basis Function (recommended), CT: Clough-Tocher, Gaussian: Post-smoothing"
    )
    smoothness = st.sidebar.slider(
        "Smoothness Parameter",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Higher values = more smoothing"
    )

    if plot_type == "3D Surface":
        plot_smoothed_iv_surface(df_iv_curve, smoothing_method=smoothing_method, smoothness=smoothness)
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

if __name__ == "__main__":
    render()