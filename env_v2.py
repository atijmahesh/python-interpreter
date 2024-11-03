# The EnvironmentManager class keeps a mapping between each variable (aka symbol)
# in a brewin program and the value of that variable - the value that's passed in can be
# anything you like. In our implementation we pass in a Value object which holds a type
# and a value (e.g., Int, 10).
class EnvironmentManager:
    # CITATION: Used ChatGPT to help me code these 18 lines.
    # Was lost on how to do stack ops and global variable management
    def __init__(self):
        self.environment_stack = [{}] 
        self.function_scope_indices = [0] 
    def push(self):
        self.environment_stack.append({})
    def pop(self):
        self.environment_stack.pop()
    def push_function_scope(self):
        self.environment_stack.append({})
        self.function_scope_indices.append(len(self.environment_stack) - 1)
    def pop_function_scope(self):
        self.environment_stack.pop()
        self.function_scope_indices.pop()
    def get(self, symbol):
        start_index = self.function_scope_indices[-1]
        for env in reversed(self.environment_stack[start_index:]):
            if symbol in env:
                return env[symbol]
        return None
    
    def set(self, symbol, value):
        start_index = self.function_scope_indices[-1]
        for env in reversed(self.environment_stack[start_index:]):
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