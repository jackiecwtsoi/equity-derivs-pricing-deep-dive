from abc import abstractmethod
from scipy.optimize import brentq, brenth
from pricing import BaseModel

class CalculatorBase:
    @abstractmethod
    def solve_for_iv(self, S, K, T, r, q, price, option_type):
        pass

class BrentIVCalculator(CalculatorBase):
    def __init__(self, pricing_model: BaseModel, vol_min, vol_max):
        self.pricing_model = pricing_model
        self.vol_min = vol_min
        self.vol_max = vol_max
    
    def solve_for_iv(self, S, K, T, r, q, price, option_type):
        if T <= 0 or K <= 0 or S <= 0 or price <= 0:
            return float('nan')
        
        def objective(sigma):
            return float(self.pricing_model.price(S, K, T, r, sigma, q, option_type) - price)
        
        try:
            iv = brentq(objective, self.vol_min, self.vol_max, maxiter=100)
            return float(iv)
        except (ValueError, RuntimeError):
            return float('nan')