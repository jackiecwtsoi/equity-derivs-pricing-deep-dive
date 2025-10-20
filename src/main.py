import streamlit as st
from exploration import black_scholes, delta, solve_for_implied_volatility

st.title("Options Pricing Learning")


S = 100.0
K = 110.0
T = 0.8
r = 0.05
sigma = 0.2
q = 0.0

call = black_scholes(S, K, T, r, sigma, q, "call")
put = black_scholes(S, K, T, r, sigma, q, "put")

print(call)
print(put)

print(delta(S, K, T, r, sigma, q, "call"))

iv = solve_for_implied_volatility(S, K, T, r, call, q, sigma_guess=1.8, option_type="call")
print(f"Implied Volatility for Call Option: {iv}")