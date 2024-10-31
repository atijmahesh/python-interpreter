# The EnvironmentManager class keeps a mapping between each variable (aka symbol)
# in a brewin program and the value of that variable - the value that's passed in can be
# anything you like. In our implementation we pass in a Value object which holds a type
# and a value (e.g., Int, 10).
class EnvironmentManager:
    # CITATION: Used ChatGPT to help me code these 16 lines.
    # Was lost on how to do stack ops and global variable management
    def __init__(self):
        self.environment_stack = [{}] 
    def push(self):
        self.environment_stack.append({})
    def pop(self):
        self.environment_stack.pop()
    def get(self, symbol):
        for env in reversed(self.environment_stack):
            if symbol in env:
                return env[symbol]
        return None
    def set(self, symbol, value):
        for env in reversed(self.environment_stack):
            if symbol in env:
                env[symbol] = value
                return True
        return False

    def create(self, symbol, start_val):
        env = self.environment_stack[-1] 
        if symbol not in env:
            env[symbol] = start_val
            return True
        return False