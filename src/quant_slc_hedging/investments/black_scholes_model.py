import numpy as np 
from dataclasses import dataclass
from scipy.stats import norm
from typing import Tuple

# Return BS option prices under input return, underlying, vols

@dataclass
class BlackScholesModelInput:
    risk_free_rate: float 
    vol: float


class BlackScholesModel:
    def __init__(self, config: BlackScholesModelInput) -> None:
        self.config = config

    def _calc_d1d2(self, underlying: np.ndarray, strike: float, time_to_exp: float) -> Tuple[np.ndarray, np.ndarray]:
        d1 = (np.log(underlying / strike) + (self.config.risk_free_rate + 0.5*self.config.vol**2) * time_to_exp) / (self.config.vol * time_to_exp**0.5)
        d2 = d1 - self.config.vol * time_to_exp**0.5
        
        return d1, d2

    def call_price(self, underlying: np.ndarray, strike: float, time_to_exp: float) -> np.ndarray:
        d1, d2 = self._calc_d1d2(underlying, strike, time_to_exp)
        return underlying * norm.cdf(d1) - strike * np.exp(-self.config.risk_free_rate * time_to_exp) * norm.cdf(d2)

    def put_price(self, underlying: np.ndarray, strike: float, time_to_exp: float) -> np.ndarray:
        d1, d2 = self._calc_d1d2(underlying, strike, time_to_exp)
        return strike * np.exp(-self.config.risk_free_rate * time_to_exp) * norm.cdf(-d2) - underlying * norm.cdf(-d1) 
    

# if __name__ == '__main__':
#     config = BlackScholesModelInput(
#         risk_free_rate=0.07,
#         vol=0.11
#     )
#     underlyings = np.linspace(100, 500, 10000) # shape of input sim array
#     bs = BlackScholesModel(config)
#     call_p = bs.call_price(underlyings, 110, 1/12)
#     print("done")