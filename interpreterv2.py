from env_v2 import EnvironmentManager
from type_valuev2 import Type, Value, create_value, get_printable
from intbase import InterpreterBase, ErrorType
from brewparse import parse_program

# handling function exceptions
class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">=", "&&", "||"}
    UNARY_OPS = {"neg", "!"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        main_func = self.__get_func_by_name("main", 0)
        self.env = EnvironmentManager()
        self.__run_statements(main_func.get("statements"))

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            param_asts = func_def.get("args")
            param_count = len(param_asts) if param_asts else 0
            func_key = (func_name, param_count)
            if func_key in self.func_name_to_ast:
                super().error(ErrorType.NAME_ERROR, f"Duplicate function error")
            self.func_name_to_ast[func_key] = func_def

    def __get_func_by_name(self, name, param_count):
        func_key = (name, param_count)
        if func_key not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        return self.func_name_to_ast[func_key]


    def __run_statements(self, statements):
        for statement in statements:
            if self.trace_output:
                print(statement)
            if statement.elem_type == InterpreterBase.FCALL_NODE:
                self.__call_func(statement)
            elif statement.elem_type == "=":
                self.__assign(statement)
            elif statement.elem_type == InterpreterBase.VAR_DEF_NODE:
                self.__var_def(statement)
            elif statement.elem_type == InterpreterBase.RETURN_NODE:
                expr = statement.get("expression")
                if expr is not None:
                    value = self.__eval_expr(expr)
                else:
                    value = Value(Type.NIL, None)  # Return NIL if no expression
                raise ReturnException(value)
            elif statement.elem_type == InterpreterBase.IF_NODE:
                self.__handle_if(statement)
            elif statement.elem_type == InterpreterBase.FOR_NODE:
                self.__handle_for(statement)

    def __handle_if(self, if_ast):
        condition_expr = if_ast.get("condition")
        condition_value = self.__eval_expr(condition_expr)
        if condition_value.type() != Type.BOOL:
            super().error(ErrorType.TYPE_ERROR, "If statement doesn't evaluate to a bool")
        if condition_value.value():
            try:
                self.env.push()
                self.__run_statements(if_ast.get("statements"))
            except ReturnException as e:
                self.env.pop()
                raise e
            self.env.pop()
        else:
            else_block = if_ast.get("else_statements")
            if else_block is not None:
                try:
                    self.env.push()
                    self.__run_statements(else_block)
                except ReturnException as e:
                    self.env.pop()
                    raise e
                self.env.pop()

    # CITATION: CHAT GPT helped me with this function (roughly 15 lines)
    def __handle_for(self, for_ast):
        init_stmt = for_ast.get("init")
        if init_stmt is not None:
            self.__assign(init_stmt)
        condition_expr = for_ast.get("condition")
        update_stmt = for_ast.get("update")
        body_statements = for_ast.get("statements")
        while True:
            condition_value = self.__eval_expr(condition_expr)
            if condition_value.type() != Type.BOOL:
                super().error(ErrorType.TYPE_ERROR, "Condition of for loop does not evaluate to a boolean")
            if not condition_value.value():
                break
            try:
                self.env.push()
                self.__run_statements(body_statements)
            except ReturnException as e:
                self.env.pop()
                raise e
            self.env.pop()
            if update_stmt is not None:
                self.__assign(update_stmt)

    def __call_func(self, call_node):
        func_name = call_node.get("name")
        if func_name == "print":
            return self.__call_print(call_node)
        if func_name == "inputi":
            return self.__call_inputi(call_node)
        if func_name == "inputs":
            return self.__call_inputs(call_node)

        args = call_node.get("args") or []
        params = len(args)
        if (func_name, params) in self.func_name_to_ast:
            return self.__execute_function(func_name, args)
        else:
            super().error(ErrorType.NAME_ERROR, f"Function {func_name} not found")

    # CITATION: Chat GPT helped me write this function (16 lines)
    def __execute_function(self, func_name, args):
        param_count = len(args)
        func_def = self.__get_func_by_name(func_name, param_count)
        param_asts = func_def.get("args")
        param_names = [param.get("name") for param in param_asts] if param_asts else []

        arg_values = [self.__eval_expr(arg) for arg in args]
        self.env.push_function_scope()
        for param_name, arg_value in zip(param_names, arg_values):
            self.env.create(param_name, arg_value)
        try:
            self.__run_statements(func_def.get("statements"))
        except ReturnException as e:
            self.env.pop_function_scope() 
            return e.value
        self.env.pop_function_scope()
        return Value(Type.NIL, None)  # Return NIL if no return value

    def __call_print(self, call_ast):
        output = ""
        for arg in call_ast.get("args"):
            result = self.__eval_expr(arg)  # result is a Value object
            output += get_printable(result)
        super().output(output)
        return Value(Type.NIL, None)

    def __call_inputi(self, call_ast):
        args = call_ast.get("args") or []
        if len(args) == 1:
            result = self.__eval_expr(args[0])
            super().output(get_printable(result))
        elif len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        inp = super().get_input()
        return Value(Type.INT, int(inp))

    def __call_inputs(self, call_ast):
        args = call_ast.get("args") or []
        if len(args) == 1:
            result = self.__eval_expr(args[0])
            super().output(get_printable(result))
        elif len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputs() function that takes > 1 parameter"
            )
        inp = super().get_input()
        return Value(Type.STRING, inp)

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = self.__eval_expr(assign_ast.get("expression"))
        if not self.env.set(var_name, value_obj):
            super().error(
                ErrorType.NAME_ERROR, f"Undefined variable {var_name} in assignment"
            )

    def __var_def(self, var_ast):
        var_name = var_ast.get("name")
        # Initialize variable with NIL
        if not self.env.create(var_name, Value(Type.NIL, None)):
            super().error(
                ErrorType.NAME_ERROR, f"Duplicate definition error"
            )

    def __eval_expr(self, expr_ast):
        if expr_ast is None:
            super().error(ErrorType.TYPE_ERROR, "None Expression")
        if expr_ast.elem_type == InterpreterBase.INT_NODE:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_NODE:
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_NODE:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.NIL_NODE:
            return Value(Type.NIL, None)
        if expr_ast.elem_type == InterpreterBase.VAR_NODE:
            var_name = expr_ast.get("name")
            val = self.env.get(var_name)
            if val is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_NODE:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type in Interpreter.UNARY_OPS:
            return self.__eval_unary_op(expr_ast)
        super().error(
            ErrorType.TYPE_ERROR, f"Unknown expression type: {expr_ast.elem_type}"
        )

    def __eval_op(self, arith_ast):
        op1 = arith_ast.get("op1")
        op2 = arith_ast.get("op2")
        if op1 is None or op2 is None:
            super().error(ErrorType.TYPE_ERROR, "Missing operands for binary operator")
        left_value_obj = self.__eval_expr(op1)
        right_value_obj = self.__eval_expr(op2)
        if left_value_obj is None or right_value_obj is None:
            super().error(ErrorType.TYPE_ERROR, "Operand evals to None")
        op = arith_ast.elem_type

        if op in ["==", "!="]:
            if left_value_obj.type() != right_value_obj.type():
                result = (op == "!=")
                return Value(Type.BOOL, result)
            else:
                if op not in self.op_to_lambda[left_value_obj.type()]:
                    super().error(
                        ErrorType.TYPE_ERROR,
                        f"Incompatible operator {op} for type {left_value_obj.type()}",
                    )
                f = self.op_to_lambda[left_value_obj.type()][op]
                return f(left_value_obj, right_value_obj)
        else:
            if left_value_obj.type() != right_value_obj.type():
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible types for {op} operation",
                )
            if op not in self.op_to_lambda.get(left_value_obj.type(), {}):
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible operator {op} for type {left_value_obj.type()}",
                )
            f = self.op_to_lambda[left_value_obj.type()][op]
            return f(left_value_obj, right_value_obj)

    def __eval_unary_op(self, unary_ast):
        operand = unary_ast.get("op1") 
        if operand is None:
            super().error(ErrorType.TYPE_ERROR, "Missing operand for unary operator")
        operand_value_obj = self.__eval_expr(operand)
        if operand_value_obj is None:
            super().error(ErrorType.TYPE_ERROR, "Operand evals to None")
        op = unary_ast.elem_type
        if op not in self.unary_op_to_lambda.get(operand_value_obj.type(), {}):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible unary operator {op} for type {operand_value_obj.type()}",
            )
        f = self.unary_op_to_lambda[operand_value_obj.type()][op]
        return f(operand_value_obj)

    def __setup_ops(self):
        self.op_to_lambda = {}
        self.unary_op_to_lambda = {}

        # set up operations on integers
        self.op_to_lambda[Type.INT] = {
            "+": lambda x, y: Value(Type.INT, x.value() + y.value()),
            "-": lambda x, y: Value(Type.INT, x.value() - y.value()),
            "*": lambda x, y: Value(Type.INT, x.value() * y.value()),
            "/": lambda x, y: Value(Type.INT, x.value() // y.value()),
            "<": lambda x, y: Value(Type.BOOL, x.value() < y.value()),
            "<=": lambda x, y: Value(Type.BOOL, x.value() <= y.value()),
            ">": lambda x, y: Value(Type.BOOL, x.value() > y.value()),
            ">=": lambda x, y: Value(Type.BOOL, x.value() >= y.value()),
            "==": lambda x, y: Value(Type.BOOL, x.value() == y.value()),
            "!=": lambda x, y: Value(Type.BOOL, x.value() != y.value()),
        }

        # set up unary operations on ints
        self.unary_op_to_lambda[Type.INT] = {
            "neg": lambda x: Value(Type.INT, -x.value()),
        }

        # set up ops on bools
        self.op_to_lambda[Type.BOOL] = {
            "&&": lambda x, y: Value(Type.BOOL, x.value() and y.value()),
            "||": lambda x, y: Value(Type.BOOL, x.value() or y.value()),
            "==": lambda x, y: Value(Type.BOOL, x.value() == y.value()),
            "!=": lambda x, y: Value(Type.BOOL, x.value() != y.value()),
        }

        # unary ops on bools
        self.unary_op_to_lambda[Type.BOOL] = {
            "!": lambda x: Value(Type.BOOL, not x.value()),
        }

        # string ops
        self.op_to_lambda[Type.STRING] = {
            "+": lambda x, y: Value(Type.STRING, x.value() + y.value()),
            "==": lambda x, y: Value(Type.BOOL, x.value() == y.value()),
            "!=": lambda x, y: Value(Type.BOOL, x.value() != y.value()),
        }

        # nil ops
        self.op_to_lambda[Type.NIL] = {
            "==": lambda x, y: Value(Type.BOOL, True),
            "!=": lambda x, y: Value(Type.BOOL, False),
        }