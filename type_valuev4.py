from intbase import InterpreterBase
from enum import Enum


class Type(Enum):
    NIL = 1
    INT = 2
    BOOL = 3
    STRING = 4

# Represents a value, which has a type and its value
class Value:
    def __init__(self, type, value=None):
        self.t = type
        self.v = value

    def value(self):
        return self.v

    def type(self):
        return self.t

    def __str__(self):
        return str(self._value)


def create_value(val):
    if val == "nil":
        return Value(Type.NIL, None)
    elif val == "true":
        return Value(Type.BOOL, True)
    elif val == "false":
        return Value(Type.BOOL, False)
    else:
        try:
            int_value = int(val)
            return Value(Type.INT, int_value)
        except ValueError:
            return Value(Type.STRING, val)


def get_printable(value_obj):
    if value_obj.type() == Type.BOOL:
        return "true" if value_obj.value() else "false"
    elif value_obj.type() == Type.NIL:
        return "nil"
    else:
        return str(value_obj.value())


class DeferredValue:
    def __init__(self, expr_ast, env_stack_copy, interpreter):
        self.expr_ast = expr_ast
        self.env_stack_copy = env_stack_copy
        self.interpreter = interpreter
        self.cached_value = None

    # CITATION: CHAT GPT HELPED ME WRITE THIS FUNCTION (9 LINES)
    def evaluate(self):
        if self.cached_value is None:
            original_env_stack = self.interpreter.env.env_stack
            self.interpreter.env.env_stack = self.env_stack_copy
            try:
                self.cached_value = self.interpreter._eval_expr_actual(self.expr_ast)
            finally:
                self.interpreter.env.env_stack = original_env_stack
        return self.cached_value
    # END CITATION
    
    def __str__(self):
        return str(self.evaluate())
