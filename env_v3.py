# The EnvironmentManager class keeps a mapping between each variable name (aka symbol)
# in a brewin program and the Value object, which stores a type, and a value.
class EnvironmentManager:
    def __init__(self):
        self.environment = []

    # returns a VariableDef object
    def get(self, symbol):
        if not self.environment:
            return None
        cur_func_env = self.environment[-1]
        for env in reversed(cur_func_env):
            if symbol in env:
                return env[symbol]
        return None

    def set(self, symbol, value):
        if not self.environment:
            return False
        cur_func_env = self.environment[-1]
        for env in reversed(cur_func_env):
            if symbol in env:
                env[symbol]["value"] = value
                return True
        return False

    # create a new symbol in the top-most environment with given val and type
    def create(self, symbol, value, var_type):
        if not self.environment:
            return False
        cur_func_env = self.environment[-1]
        if symbol in cur_func_env[-1]:   # symbol already defined in current scope
            return False
        cur_func_env[-1][symbol] = {"value": value, "type": var_type}
        return True

    # used when we enter a new function - start with empty dictionary to hold parameters.
    def push_func(self):
        self.environment.append([{}])  # [[...]] -> [[...], [{}]]

    def push_block(self):
        if not self.environment:
            return
        cur_func_env = self.environment[-1]
        cur_func_env.append({})  # [[...],[{....}] -> [[...],[{...}, {}]]

    def pop_block(self):
        if not self.environment:
            return
        cur_func_env = self.environment[-1]
        if len(cur_func_env) > 1:
            cur_func_env.pop()

    # used on function exit: discard all blocks for the function
    def pop_func(self):
        if self.environment:
            self.environment.pop()
