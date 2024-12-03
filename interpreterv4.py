import copy
from enum import Enum
from brewparse import parse_program
from env_v4 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev4 import Type, Value, create_value, get_printable, DeferredValue

class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2

class BException(Exception):
    def __init__(self, exception_type):
        self.exception_type = exception_type

# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

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
        self.env = EnvironmentManager()
        try:
            self.__call_func_aux("main", [])
        except BException as e:
            super().error(ErrorType.FAULT_ERROR, f"Exception: {e.exception_type}")

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
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
        try:
            for statement in statements:
                if self.trace_output:
                    print(statement)
                status, return_val = self.__run_statement(statement)
                if status == ExecStatus.RETURN:
                    self.env.pop_block()
                    return (status, return_val)
        except BException as e:
            self.env.pop_block()
            raise e
        self.env.pop_block()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __run_statement(self, statement):
        status = ExecStatus.CONTINUE
        return_val = None
        if statement.elem_type == InterpreterBase.FCALL_NODE:
            self.__call_func(statement, eager=True)  
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
        elif statement.elem_type == InterpreterBase.RAISE_NODE:
            self.__do_raise(statement)
        elif statement.elem_type == InterpreterBase.TRY_NODE:
            status, return_val = self.__do_try(statement)
            if status == ExecStatus.RETURN:
                return (status, return_val)
        return (status, return_val)
    
    def __call_func(self, call_node, eager=False):
        func_name = call_node.get("name")
        actual_args = call_node.get("args")
        return self.__call_func_aux(func_name, actual_args, eager=eager)

    def __call_func_aux(self, func_name, actual_args, env=None, eager=False):
        if env is None:
            env = self.env
        if func_name == "print":
            return self.__call_print(actual_args)
        if func_name == "inputi" or func_name == "inputs":
            return self.__call_input(func_name, actual_args)

        func_ast = self.__get_func_by_name(func_name, len(actual_args))
        formal_args = func_ast.get("args")
        if len(actual_args) != len(formal_args):
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
            )

        # create the new activation record
        self.env.push_func()
        # and add the formal arguments to the activation record
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            arg_name = formal_ast.get("name")
            if eager:
                value_obj = self.__eval_expr(actual_ast, eager=True, env=env)
            else:
                value_obj = DeferredValue(actual_ast, env.copy_full_env(), self)
            self.env.create(arg_name, value_obj)

        try:
            status, return_val = self.__run_statements(func_ast.get("statements"))
        except BException as e:
            self.env.pop_func()
            raise e
        self.env.pop_func()
        return return_val

    def __call_print(self, args):
        output = ""
        try:
            for arg in args:
                result = self.__eval_expr(arg, eager=True)  # eager eval
                output += get_printable(result)
        except BException as e:
            raise e
        super().output(output)
        return Interpreter.NIL_VALUE

    def __call_input(self, name, args):
        if args is not None and len(args) == 1:
            result = self.__eval_expr(args[0], eager=True) # eager eval
            super().output(get_printable(result))
        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        inp = super().get_input()
        if name == "inputi":
            return Value(Type.INT, int(inp))
        if name == "inputs":
            return Value(Type.STRING, inp)

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = DeferredValue(assign_ast.get("expression"), self.env.copy_full_env(), self)
        if not self.env.set(var_name, value_obj):
            super().error(
                ErrorType.NAME_ERROR, f"Undefined variable {var_name} in assignment"
            )

    def __var_def(self, var_ast):
        var_name = var_ast.get("name")
        if not self.env.create(var_name, Interpreter.NIL_VALUE):
            super().error(
                ErrorType.NAME_ERROR, f"Duplicate definition for variable {var_name}"
            )

    def __eval_expr(self, expr_ast, eager=False, env=None):
        if env is None:
            env = self.env
        if eager:
            return self._eval_expr_actual(expr_ast, env)
        else:
            return DeferredValue(expr_ast, env.copy_full_env(), self)

    def _eval_expr_actual(self, expr_ast, env=None):
        if env is None:
            env = self.env
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
            val = env.get(var_name)
            if val is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            if isinstance(val, DeferredValue):
                val = val.evaluate()
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_NODE:
            func_name = expr_ast.get("name")
            actual_args = expr_ast.get("args")
            return self.__call_func_aux(func_name, actual_args, env, eager=False)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast, env)
        if expr_ast.elem_type == InterpreterBase.NEG_NODE:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x, env)
        if expr_ast.elem_type == InterpreterBase.NOT_NODE:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x, env)

    def __eval_op(self, arith_ast, env):
        left_value_obj = self._eval_expr_actual(arith_ast.get("op1"), env)
        if isinstance(left_value_obj, DeferredValue):
            left_value_obj = left_value_obj.evaluate()
        # making and + or short circuited
        if arith_ast.elem_type == '&&':
            if left_value_obj.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for {arith_ast.elem_type} operation",
                )
            if not left_value_obj.value():
                return Value(Type.BOOL, False)
            right_value_obj = self._eval_expr_actual(arith_ast.get("op2"), env)
            if isinstance(right_value_obj, DeferredValue):
                right_value_obj = right_value_obj.evaluate()
            if right_value_obj.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for {arith_ast.elem_type} operation",
                )
            return Value(Type.BOOL, left_value_obj.value() and right_value_obj.value())
        elif arith_ast.elem_type == '||':
            if left_value_obj.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for {arith_ast.elem_type} operation",
                )
            if left_value_obj.value():
                return Value(Type.BOOL, True)
            right_value_obj = self._eval_expr_actual(arith_ast.get("op2"), env)
            if isinstance(right_value_obj, DeferredValue):
                right_value_obj = right_value_obj.evaluate()
            if right_value_obj.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for {arith_ast.elem_type} operation",
                )
            return Value(Type.BOOL, left_value_obj.value() or right_value_obj.value())
        else:
            right_value_obj = self._eval_expr_actual(arith_ast.get("op2"), env)
            if isinstance(right_value_obj, DeferredValue):
                right_value_obj = right_value_obj.evaluate()
            if not self.__compatible_types(
                arith_ast.elem_type, left_value_obj, right_value_obj
            ):
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible types for {arith_ast.elem_type} operation",
                )
            if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible operator {arith_ast.elem_type} for type {left_value_obj.type()}",
                )
            f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
            return f(left_value_obj, right_value_obj)

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f, env):
        value_obj = self._eval_expr_actual(arith_ast.get("op1"), env)
        if isinstance(value_obj, DeferredValue):
            value_obj = value_obj.evaluate()
        if value_obj.type() != t:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible type for {arith_ast.elem_type} operation",
            )
        return Value(t, f(value_obj.value()))

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
        self.op_to_lambda[Type.INT]["/"] = self.__int_divide
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
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
            x.type(), x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            x.type(), x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

    def __int_divide(self, x, y):
        if y.value() == 0:
            raise BException("div0")
        return Value(x.type(), x.value() // y.value())

    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        try:
            result = self.__eval_expr(cond_ast, eager=True)
        except BException as e:
            raise e
        if result.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR,
                "Incompatible type for if condition",
            )
        if result.value():
            statements = if_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            return (status, return_val)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                status, return_val = self.__run_statements(else_statements)
                return (status, return_val)

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_for(self, for_ast):
        init_ast = for_ast.get("init") 
        cond_ast = for_ast.get("condition")
        update_ast = for_ast.get("update") 

        self.__run_statement(init_ast)  # initialize counter variable
        while True:
            try:
                run_for = self.__eval_expr(cond_ast, eager=True)
            except BException as e:
                raise e
            if run_for.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for for condition",
                )
            if not run_for.value():
                break
            statements = for_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            if status == ExecStatus.RETURN:
                return status, return_val
            self.__run_statement(update_ast)  # update counter variable

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return (ExecStatus.RETURN, Interpreter.NIL_VALUE)
        value_obj = self._eval_expr_actual(expr_ast)
        return (ExecStatus.RETURN, value_obj)

    def __do_raise(self, raise_ast):
        expr_ast = raise_ast.get("exception_type")
        exception_value = self.__eval_expr(expr_ast, eager=True)
        if exception_value.type() != Type.STRING:
            super().error(ErrorType.TYPE_ERROR, "Exception message must be a string")
        raise BException(exception_value.value())

    # CITATION: I used ChatGPT to help me write this function (17 lines)
    def __do_try(self, try_ast):
        try:
            self.env.push_block()
            status, return_val = self.__run_statements(try_ast.get("statements"))
            self.env.pop_block()
            return (status, return_val)
        except BException as e:
            self.env.pop_block()
            catchers = try_ast.get("catchers")
            for catch_ast in catchers:
                if catch_ast.get("exception_type") == e.exception_type:
                    self.env.push_block()
                    status, return_val = self.__run_statements(catch_ast.get("statements"))
                    self.env.pop_block()
                    return (status, return_val)
            raise e
    # END CITATION
