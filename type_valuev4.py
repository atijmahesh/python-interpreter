from intbase import InterpreterBase


# Enumerated type for our different language data types
class Type:
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    NIL = "nil"


# Represents a value, which has a type and its value
class Value:
    def __init__(self, type, value=None):
        self.t = type
        self.v = value

    def value(self):
        return self.v

    def type(self):
        return self.t


def create_value(val):
    if val == InterpreterBase.TRUE_DEF:
        return Value(Type.BOOL, True)
    elif val == InterpreterBase.FALSE_DEF:
        return Value(Type.BOOL, False)
    elif val == InterpreterBase.NIL_DEF:
        return Value(Type.NIL, None)
    elif isinstance(val, str):
        return Value(Type.STRING, val)
    elif isinstance(val, int):
        return Value(Type.INT, val)
    else:
        raise ValueError("Unknown value type")


def get_printable(val):
    if val.type() == Type.INT:
        return str(val.value())
    if val.type() == Type.STRING:
        return val.value()
    if val.type() == Type.BOOL:
        if val.value() is True:
            return "true"
        return "false"
    return None

class DeferredValue:
    def __init__(self, expr_ast, env_manager, interpreter):
        self.expr_ast = expr_ast
        self.env = env_manager.copy_current_env()
        self.interpreter = interpreter
        self.cached_value = None

    def evaluate(self):
        if self.cached_value is None:
            original_env = self.interpreter.env
            self.interpreter.env = self.env
            try:
                self.cached_value = self.interpreter._eval_expr_actual(self.expr_ast)
            finally:
                self.interpreter.env = original_env
        return self.cached_value
