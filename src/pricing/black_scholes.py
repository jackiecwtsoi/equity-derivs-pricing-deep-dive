import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from .base import BaseModel

class BlackScholesModel(BaseModel):
    def __init__(self):
        pass

    @staticmethod
    def price(S, K, T, r, sigma, q, option_type) -> float:
        d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)

        if (option_type == "call"):
            call = (S * jnp.exp(-q * T) * jnorm.cdf(d1)) - (K * jnp.exp(-r * T) * jnorm.cdf(d2))                
            return call
        else:
            put = (K * jnp.exp(-r * T) * jnorm.cdf(-d2) - (S * jnp.exp(-q * T) * jnorm.cdf(-d1)))
            return put