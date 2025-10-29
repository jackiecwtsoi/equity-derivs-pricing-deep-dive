import jax.numpy as jnp
from jax import grad
from jax.scipy.stats import norm as jnorm
from scipy.optimize import newton, brentq

def black_scholes(S, K, T, r, sigma, q=0, option_type="call"):
    """
    Calculate the Black-Scholes option price.
    
    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration in years
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying asset
    q : float : Dividend yield (default is 0)
    option_type : str : 'call' for call option, 'put' for put option
    
    Returns:
    float : Option price
    """
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    if (option_type == "call"):
        call = (S * jnp.exp(-q * T) * jnorm.cdf(d1)) - (K * jnp.exp(-r * T) * jnorm.cdf(d2))
        return call
    else:
        put = (K * jnp.exp(-r * T) * jnorm.cdf(-d2) - (S * jnp.exp(-q * T) * jnorm.cdf(-d1)))
        return put

delta = grad(black_scholes, argnums=0)
rho = grad(black_scholes, 3)
vega = grad(black_scholes, 4)


def loss(S, K, T, r, sigma_guess, price, q=0, option_type="call"):
    # 1. Calculate the option price with a GUESS for the volatility
    theoretical_price = black_scholes(S, K, T, r, sigma_guess, q, option_type)

    # 2. Calculate the loss between the theoretical price and market price
    return theoretical_price - price

loss_grad = grad(loss, argnums=4)

def solve_for_implied_volatility(S, K, T, r, price, q=0, sigma_guess=0.8, option_type="call",
                               n_iter=50, epsilon=1e-6, verbose=False):
    """
    Solve for implied volatility using scipy.optimize.newton
    Returns float('nan') if no solution found
    """
    # Input validation
    if T <= 0 or K <= 0 or S <= 0 or not jnp.isfinite(price):
        return float('nan')
    
    def objective(sigma):
        return float(black_scholes(S, K, T, r, sigma, q, option_type) - price)
    
    def objective_prime(sigma):
        return float(loss_grad(S, K, T, r, sigma, price, q, option_type))
    
    try:
        result = newton(
            func=objective,
            x0=sigma_guess,
            fprime=objective_prime,
            tol=epsilon,
            maxiter=n_iter,
            full_output=verbose
        )
        
        if verbose:
            print(f"Convergence info: {result}")
            
        # If full_output=True, result is a tuple and we want first element
        sigma = result[0] if verbose else result
        
        # Validate result
        if not jnp.isfinite(sigma) or sigma <= 0:
            return float('nan')
            
        return float(sigma)
        
    except Exception as e:
        if verbose:
            print(f"Newton method failed: {e}")
        return float('nan')

def solve_for_iv_brent(S, K, T, r, price, q=0, option_type="call", vol_min=0.0001, vol_max=5.0):
    if T <= 0 or K <= 0 or S <= 0 or price <= 0:
        return float('nan')
    
    def objective(sigma):
        return float(black_scholes(S, K, T, r, sigma, q, option_type) - price)
    
    try:
        iv = brentq(objective, vol_min, vol_max, maxiter=100)
        return float(iv)
    except (ValueError, RuntimeError):
        return float('nan')
    