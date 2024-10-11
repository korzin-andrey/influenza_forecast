class BRModel:
    """
    Interface for all models
    """
    Br_func_array = [0.1, 0.1, 1, 0.9, 0.55, 0.3, 0.15, 0.05]

    def __init__():
        ...

    def simulate():
        ...

    def __br_function(self, day: int):
        if day >= len(self.Br_func_array):
            return 0
        return self.Br_func_array[day]