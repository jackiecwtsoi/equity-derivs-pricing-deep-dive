import jax.numpy as jnp
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
        # TODO - Need to fix this to calculate multi-step scenario also
        prices = self.__generate_risk_neutral_price_paths_single_step(S, T, r, sigma, q)
        print(prices)

        # 2. Calculate option payoff for each path
        payoffs = jnp.maximum(prices - K, 0)
        print(payoffs)

        # 3. Average all payoffs and discount to present value
        option_price = jnp.exp(-r * T) * jnp.mean(payoffs)

        return option_price
    
    
    def __generate_risk_neutral_price_paths_single_step(self, S0, T, r, sigma, q):
        # 1. Generate standard normal random shocks (effectively z(t))
        Z = random.normal(random.PRNGKey(0), shape=(self.num_simulations,))

        # 2. Simulate prices at maturity directly (for European options as this is the only thing we care about)
        final_prices_at_maturity = S0 * jnp.exp((r - q - 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * Z)

        return final_prices_at_maturity
