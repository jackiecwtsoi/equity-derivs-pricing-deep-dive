import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

class VolatilitySurface:
    def __init__(self, df: pd.DataFrame,
                 moneyness_col: str = "Moneyness",
                 time_to_maturity_col: str = "Time to Maturity",
                 iv_col: str = "Calculated IV"):
        self.df = df.dropna(subset=[moneyness_col, time_to_maturity_col, iv_col])
        self.moneyness_col = moneyness_col
        self.time_to_maturity_col = time_to_maturity_col
        self.iv_col = iv_col

    def _set_smoothing_method(self, smoothing_method: str):
        if smoothing_method == 'rbf':
            pass
        elif smoothing_method == 'gaussian':
            pass

    def build_surface(self):
        pass

    def interpolate_iv(self, moneyness: float, time_to_maturity: float) -> float:
        pass