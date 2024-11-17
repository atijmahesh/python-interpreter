from intbase import InterpreterBase

# Enumerated type for our different language data types
class Type:
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    NIL = "nil"
    VOID = "void"

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
    if val.type() == Type.NIL:
        return "nil"
    return None

def default_value_for_type(t):
    # Provide a default value for each type
    if t == Type.INT:
        return Value(Type.INT, 0)
    if t == Type.BOOL:
        return Value(Type.BOOL, False)
    if t == Type.STRING:
        return Value(Type.STRING, "")
    if t == Type.NIL:
        return Value(Type.NIL, None)
    # If type is void or unknown, return nil
    return Value(Type.NIL, None)
