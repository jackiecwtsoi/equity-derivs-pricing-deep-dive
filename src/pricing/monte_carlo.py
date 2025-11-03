import jax.numpy as jnp
import numpy as np
from jax import random
from .base import BaseModel


'''
Monte Carlo simulation
- Simulates multiple price paths for the underlying asset
- Returns the present value of the expected payoff
'''
class MonteCarloModel(BaseModel):
    def __init__(self, num_simulations: int):
        self.num_simulations = num_simulations

    def price(self, S, K, T, r, sigma, q, option_type) -> float:
        # 1. Generate random price paths
        price_matrix = self.__generate_risk_neutral_price_paths_multi_step(S, T, r, sigma, q, 100)
        prices = price_matrix[:, -1] # Last column for European options (FIXME)

        # prices = self.__generate_risk_neutral_price_paths_single_step(S, T, r, sigma, q)

        # 2. Calculate option payoff for each path
        payoffs = jnp.maximum(prices - K, 0)

        # 3. Average all payoffs and discount to present value
        option_price = jnp.exp(-r * T) * jnp.mean(payoffs)

        return option_price
    
    @DeprecationWarning
    def __generate_risk_neutral_price_paths_single_step(self, S0, T, r, sigma, q):
        # 1. Generate standard normal random shocks (effectively z(t))
        Z = random.normal(random.PRNGKey(0), shape=(self.num_simulations,))

        # 2. Simulate prices at maturity directly (for European options as this is the only thing we care about)
        final_prices_at_maturity = S0 * jnp.exp((r - q - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * Z)

        return final_prices_at_maturity

    def __generate_risk_neutral_price_paths_multi_step(self, S0, T, r, sigma, q, num_steps):
        dt = T / num_steps

        # 1. Initialize price matrix of dimension num_simulations * (num_steps + 1)
        price_matrix = np.zeros((self.num_simulations, num_steps + 1))
        price_matrix[:, 0] = S0 # All paths start at S0

        # 2. Generate standard normal random shocks for all steps
        Z = np.random.standard_normal((self.num_simulations, num_steps))

        # 3. Simulate prices at each time step
        for t in range(1, num_steps + 1):
            price_matrix[:, t] = price_matrix[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
        
        return price_matrix