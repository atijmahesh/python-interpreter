import copy
from enum import Enum

from brewparse import parse_program
from env_v3 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev3 import Type, Value, create_value, get_printable, default_value_for_type


class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    FALSE_VALUE = create_value(InterpreterBase.FALSE_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()
        self.func_name_to_ast = {}

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        self.env = EnvironmentManager()
        self.__call_func_aux("main", [])

    def __set_up_function_table(self, ast):
        for func_def in ast.get("functions", []):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
            return_type = func_def.get("return_type")
            if return_type is None:
                return_type = InterpreterBase.VOID_DEF
                func_def.dict["return_type"] = return_type
            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = {}
            self.func_name_to_ast[func_name][num_params] = func_def

    def __get_func_by_name(self, name, num_params):
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        candidate_funcs = self.func_name_to_ast[name]
        if num_params not in candidate_funcs:
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {name} taking {num_params} params not found",
            )
        return candidate_funcs[num_params]

    def __run_statements(self, statements):
        self.env.push_block()
        for statement in statements:
            if self.trace_output:
                print(statement)
            status, return_val = self.__run_statement(statement)
            if status == ExecStatus.RETURN:
                self.env.pop_block()
                return status, return_val
        self.env.pop_block()
        return ExecStatus.CONTINUE, Interpreter.NIL_VALUE

    def __run_statement(self, statement):
        status = ExecStatus.CONTINUE
        return_val = None
        if statement.elem_type == InterpreterBase.FCALL_NODE:
            return_val = self.__call_func(statement)
        elif statement.elem_type == "=":
            self.__assign(statement)
        elif statement.elem_type == InterpreterBase.VAR_DEF_NODE:
            self.__var_def(statement)
        elif statement.elem_type == InterpreterBase.RETURN_NODE:
            status, return_val = self.__do_return(statement)
        elif statement.elem_type == InterpreterBase.IF_NODE:
            status, return_val = self.__do_if(statement)
        elif statement.elem_type == InterpreterBase.FOR_NODE:
            status, return_val = self.__do_for(statement)
        return status, return_val

    def __call_func(self, call_node):
        func_name = call_node.get("name")
        actual_args = call_node.get("args", [])
        return self.__call_func_aux(func_name, actual_args)

    def __call_func_aux(self, func_name, actual_args):
        if func_name == "print":
            return self.__call_print(actual_args)
        if func_name == "inputi" or func_name == "inputs":
            return self.__call_input(func_name, actual_args)

        num_params = len(actual_args) if actual_args else 0
        func_ast = self.__get_func_by_name(func_name, num_params)
        formal_args = func_ast.get("args", [])
        return_type = func_ast.get("return_type", InterpreterBase.VOID_DEF)
        # Eval params
        evaluated_args = []
        for i, arg_ast in enumerate(actual_args):
            evaluated_arg = self.__eval_expr(arg_ast)
            param_info = formal_args[i]
            param_name = param_info.get("name")
            param_type = param_info.get("var_type")
            if param_type is None:
                super().error(ErrorType.TYPE_ERROR, f"No type specified for parameter '{param_name}'!'")
            if not self.__check_type_compatibility(param_type, evaluated_arg.type()):
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible argument type for parameter '{param_name}'. Expected '{param_type}', got '{evaluated_arg.type()}'"
                )
            evaluated_args.append((param_name, param_type, evaluated_arg))
        self.env.push_func()
        # CITATION, CHAT GPT Helped me write these 4 lines on enviornment updates
        for param_name, param_type, param_value in evaluated_args:
            if not self.env.create(param_name, param_value, param_type):
                super().error(
                    ErrorType.NAME_ERROR, f"Duplicate definition for parameter {param_name}"
                )
        # END CITATION
        
        status, return_val = self.__run_statements(func_ast.get("statements"))
        self.env.pop_func()
        if status != ExecStatus.RETURN and return_type != InterpreterBase.VOID_DEF:
            return_val = default_value_for_type(return_type)
        return return_val

    def __call_print(self, args):
        output_parts = []
        for arg in args:
            result = self.__eval_expr(arg)  # result is a Value object
            output_parts.append(get_printable(result))
        output = "".join(output_parts)
        super().output(output)
        return Interpreter.NIL_VALUE

    def __call_input(self, name, args):
        if len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        if len(args) == 1:
            result = self.__eval_expr(args[0])
            super().output(get_printable(result))
        inp = super().get_input()
        if inp is None:
            inp = ""
        if name == "inputi":
            # Convert ip to int if possible
            try:
                val = int(inp)
            except ValueError:
                super().error(ErrorType.TYPE_ERROR, "Non-integer input for inputi()")
            return Value(Type.INT, val)
        if name == "inputs":
            return Value(Type.STRING, inp)

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = self.__eval_expr(assign_ast.get("expression"))
        var_def = self.env.get(var_name)
        if var_def is None:
            super().error(
                ErrorType.NAME_ERROR, f"Undefined variable {var_name} in assignment"
            )
        var_type = var_def["type"]
        if not self.__check_type_compatibility(var_type, value_obj.type()):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types, '{var_name}' is type '{var_type}' and not '{value_obj.type()}'"
            )
        if not self.env.set(var_name, value_obj):
            super().error(
                ErrorType.NAME_ERROR, f"Undefined variable {var_name} in assignment"
            )

    def __var_def(self, var_ast):
        var_name = var_ast.get("name")
        var_type = var_ast.get("var_type")
        if var_type is None:
            super().error(ErrorType.TYPE_ERROR, f"No type specified for variable '{var_name}'")
        # Check if type is valid
        if not self.__is_valid_type(var_type):
            super().error(ErrorType.TYPE_ERROR, f"Invalid type for variable '{var_name}'")
        # Initialize the variable with a default val
        default_val = default_value_for_type(var_type)
        if not self.env.create(var_name, default_val, var_type):
            super().error(
                ErrorType.NAME_ERROR, f"Duplicate definition for variable {var_name}"
            )

    def __eval_expr(self, expr_ast):
        if expr_ast.elem_type == InterpreterBase.NIL_NODE:
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_NODE:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_NODE:
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_NODE:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_NODE:
            var_name = expr_ast.get("name")
            var_def = self.env.get(var_name)
            if var_def is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            return var_def["value"]
        if expr_ast.elem_type == InterpreterBase.FCALL_NODE:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type == InterpreterBase.NEG_NODE:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -x)
        if expr_ast.elem_type == InterpreterBase.NOT_NODE:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)

    def __eval_op(self, arith_ast):
        op = arith_ast.elem_type
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))
        if op in {"+", "-", "*", "/", "<", "<=", ">", ">="}:
            # Expect both operands to be ints
            if left_value_obj.type() != Type.INT or right_value_obj.type() != Type.INT:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible types for {op} operation'"
                )
        elif op in {"==", "!=", "||", "&&"}:
            if op in {"||", "&&"} and (left_value_obj.type() != Type.BOOL or right_value_obj.type() != Type.BOOL):
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible types for {op} operation"
                )
        if left_value_obj.type() not in self.op_to_lambda:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {op} operation",
            )
        if op not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {op} for type {left_value_obj.type()}",
            )
        f = self.op_to_lambda[left_value_obj.type()][op]
        return f(left_value_obj, right_value_obj)

    def __eval_unary(self, arith_ast, expected_type, operation):
        value_obj = self.__eval_expr(arith_ast.get("op1"))
        if value_obj.type() != expected_type:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible type for {arith_ast.elem_type} operation",
            )
        result_val = operation(value_obj.value())
        return Value(value_obj.type(), result_val)

    def __setup_ops(self):
        self.op_to_lambda = {}
        # set up operations on integers
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), x.value() - y.value()
        )
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), x.value() * y.value()
        )
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            x.type(), x.value() // y.value() if y.value() != 0 else 0
        )
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )
        #  set up operations on strings
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.STRING]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.STRING]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        #  set up operations on bools
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            Type.BOOL, x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            Type.BOOL, x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == Type.NIL and y.type() == Type.NIL
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, not (x.type() == Type.NIL and y.type() == Type.NIL)
        )

    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        result = self.__eval_expr(cond_ast)
        if result.type() == Type.INT:
            result = self.__coerce_type(result, Type.BOOL)
        if result.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR,
                "Incompatible type for if condition",
            )
        if result.value():
            statements = if_ast.get("statements")
            return self.__run_statements(statements)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                return self.__run_statements(else_statements)
        return ExecStatus.CONTINUE, Interpreter.NIL_VALUE

    def __do_for(self, for_ast):
        init_ast = for_ast.get("init") 
        cond_ast = for_ast.get("condition")
        update_ast = for_ast.get("update")
        self.__run_statement(init_ast)  # initialize counter variable
        while True:
            run_for_val = self.__eval_expr(cond_ast)
            if run_for_val.type() == Type.INT:
                # Coerce int to bool if needed
                run_for_val = self.__coerce_type(run_for_val, Type.BOOL)
            if run_for_val.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for for loop condition: expected bool or int, got '{run_for_val.type()}'"
                )
            # CITATION: USED CHAT GPT TO WRITE THE FOLLOwWING 7 LINES
            if not run_for_val.value():
                break
            statements = for_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            if status == ExecStatus.RETURN:
                return status, return_val
            self.__run_statement(update_ast)  # update counter variable
            # END CITATION
        return ExecStatus.CONTINUE, Interpreter.NIL_VALUE

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return ExecStatus.RETURN, Interpreter.NIL_VALUE
        value_obj = self.__eval_expr(expr_ast)
        return ExecStatus.RETURN, value_obj

    def __is_valid_type(self, t):
        return t in {Type.INT, Type.BOOL, Type.STRING, InterpreterBase.VOID_DEF}

    def __check_type_compatibility(self, target_type, source_type):
        if target_type == source_type:
            return True
        if target_type == Type.BOOL and source_type == Type.INT:
            return True
        return False

    def __coerce_type(self, value_obj, target_type):
        if value_obj.type() == target_type:
            return value_obj
        # Coerce int to bool if target is bool
        if value_obj.type() == Type.INT and target_type == Type.BOOL:
            coerced_val = (value_obj.value() != 0)
            return Value(Type.BOOL, coerced_val)
        return value_obj
