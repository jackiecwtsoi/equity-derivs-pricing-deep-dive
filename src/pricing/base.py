from abc import abstractmethod

class BaseModel():
    @abstractmethod
    def price(self, S, K, T, r, sigma, q, option_type) -> float:
        pass