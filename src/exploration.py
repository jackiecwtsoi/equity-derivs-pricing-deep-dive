import jax.numpy as jnp
from jax import grad
from jax.scipy.stats import norm as jnorm

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
                                 method="newton_rhapson", n_iter=20, epsilon=0.001, verbose=True):
    # 1. Start with an initial guess value for sigma (IV)
    sigma_candidate = sigma_guess

    # 2. Loop through until convergence / a certain number of iteractions
    for i in range(n_iter):
        # 3. Calcualte the loss value
        loss_val = loss(S, K, T, r, sigma_candidate, price, q, option_type)
        if verbose:
            print(f"Iteraction {i}")
            print(f"Current loss: {loss_val}")
            print("---------------------")

        # 4. Check if the loss value is within epsilon
        if abs(loss_val) < epsilon:
            break
        else: 
            # 5. If loss value is still not small enough, then update the new sigma guess value
            loss_grad_val = loss_grad(S, K, T, r, sigma_candidate, price, q, option_type)
            sigma_candidate = sigma_candidate - loss_val / loss_grad_val
    
    return sigma_candidate
