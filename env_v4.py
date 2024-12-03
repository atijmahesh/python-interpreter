# The EnvironmentManager class keeps a mapping between each variable name (aka symbol)
# in a brewin program and the Value object, which stores a type, and a value.
class EnvironmentManager:
    def __init__(self):
        # stack of envs
        self.env_stack = []

    def get(self, var_name):
        for env in reversed(self.env_stack):
            if var_name in env:
                return env[var_name]
        return None

    def set(self, var_name, value):
        for env in reversed(self.env_stack):
            if var_name in env:
                env[var_name] = value
                return True
        return False

    def push_func(self):
        self.env_stack.append({})
    def pop_func(self):
        self.env_stack.pop()

    def push_block(self):
        self.env_stack.append({})

    def pop_block(self):
        self.env_stack.pop()

    def create(self, var_name, value):
        if var_name in self.env_stack[-1]:
            return False
        self.env_stack[-1][var_name] = value
        return True

    def copy_full_env(self):
        return [env.copy() for env in self.env_stack]
