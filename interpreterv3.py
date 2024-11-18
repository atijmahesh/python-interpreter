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
        self.struct_defs = {}

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_struct_definitions(ast)
        self.__set_up_function_table(ast)
        self.env = EnvironmentManager()
        self.__call_func_aux("main", [])

    # add struct defs from ast and store them
    def __set_up_struct_definitions(self, ast):
        structs = ast.get("structs")
        if structs is None:
            structs = []
        # get struct names first
        for struct_def in structs:
            struct_name = struct_def.get("name")
            if struct_name in self.struct_defs:
                super().error(ErrorType.TYPE_ERROR, f"Duplicate struct '{struct_name}'")
            self.struct_defs[struct_name] = None
        for struct_def in structs:
            struct_name = struct_def.get("name")
            fields_ast = struct_def.get("fields") or []
            fields = {}
            for field_ast in fields_ast:
                field_name = field_ast.get("name")
                field_type = field_ast.get("var_type")
                if field_type is None:
                    super().error(ErrorType.TYPE_ERROR, f"No type specified for field '{field_name}'")
                if not self.__is_valid_type(field_type):
                    super().error(ErrorType.TYPE_ERROR, f"Invalid type for field '{field_name}'")
                fields[field_name] = field_type
            self.struct_defs[struct_name] = fields

    def __set_up_function_table(self, ast):
        functions = ast.get("functions")
        if functions is None:
            functions = []
        for func_def in functions:
            func_name = func_def.get("name")
            num_params = len(func_def.get("args") or [])
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
        actual_args = call_node.get("args") or []
        return self.__call_func_aux(func_name, actual_args)

    def __call_func_aux(self, func_name, actual_args):
        if func_name == "print":
            return self.__call_print(actual_args)
        if func_name == "inputi" or func_name == "inputs":
            return self.__call_input(func_name, actual_args)

        num_params = len(actual_args)
        func_ast = self.__get_func_by_name(func_name, num_params)
        formal_args = func_ast.get("args") or []
        return_type = func_ast.get("return_type") or InterpreterBase.VOID_DEF
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
            evaluated_arg = self.__coerce_type(evaluated_arg, param_type)
            evaluated_args.append((param_name, param_type, evaluated_arg))
        self.env.push_func()
        # CITATION, CHAT GPT Helped me write these 25 lines on environment updates
        for param_name, param_type, param_value in evaluated_args:
            if not self.env.create(param_name, param_value, param_type):
                super().error(
                    ErrorType.NAME_ERROR, f"Duplicate definition for parameter {param_name}"
                )
        status, return_val = self.__run_statements(func_ast.get("statements") or [])
        self.env.pop_func()
        if status == ExecStatus.RETURN:
            if return_type == InterpreterBase.VOID_DEF:
                if return_val is not None:
                    super().error(ErrorType.TYPE_ERROR, f"Function '{func_name}' should not return a value")
                return_val = Value(Type.VOID, None)
            else:
                if return_val is None:
                    return_val = default_value_for_type(return_type)
                elif not self.__check_type_compatibility(return_type, return_val.type()):
                    super().error(
                        ErrorType.TYPE_ERROR,
                        f"Incompatible return type in function '{func_name}': expected '{return_type}', got '{return_val.type()}'"
                    )
        else:
            if return_type != InterpreterBase.VOID_DEF:
                return_val = default_value_for_type(return_type)
            else:
                return_val = Value(Type.VOID, None)
        # END CITATION
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
        if '.' in var_name:
            self.__assign_to_field(var_name, value_obj)
            return
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
        # coerce if needed
        value_obj = self.__coerce_type(value_obj, var_type)
        if not self.env.set(var_name, value_obj):
            super().error(
                ErrorType.NAME_ERROR, f"Undefined variable {var_name} in assignment"
            )
    # assign var to struct field
    def __assign_to_field(self, field_path, value_obj):
        path_parts = field_path.split('.')
        base_var_name = path_parts[0]
        var_def = self.env.get(base_var_name)
        if var_def is None:
            super().error(ErrorType.NAME_ERROR, f"Undefined variable in field assignment")
        curr_val = var_def["value"]
        if curr_val.type() not in self.struct_defs and curr_val.type() != Type.NIL:
            super().error(ErrorType.TYPE_ERROR, f"Variable '{base_var_name}' is not a struct in field assignment")
        # CITATION: CHATGPT helped me write the following 23 lines
        for i in range(1, len(path_parts)):
            field_name = path_parts[i]
            if curr_val.type() == Type.NIL:
                super().error(ErrorType.FAULT_ERROR, f"Null reference on '{field_path}'")
            struct_type = curr_val.type()
            fields_def = self.struct_defs.get(struct_type, {})
            if field_name not in fields_def:
                super().error(ErrorType.NAME_ERROR, f"Field '{field_name}' not found in struct '{struct_type}'")
            fields_dict = curr_val.value()
            if i < len(path_parts) - 1:
                # Not the last field, navigate deeper
                next_value = fields_dict[field_name]
                curr_val = next_value
            else:
                field_type = fields_def[field_name]
                # Coerce value if needed
                coerced_value = self.__coerce_type(value_obj, field_type)
                if not self.__check_type_compatibility(field_type, coerced_value.type()):
                    super().error(
                        ErrorType.TYPE_ERROR,
                        f"Incompatible types for field assignment: field '{field_name}' is type '{field_type}', value is type '{coerced_value.type()}'"
                    )
                fields_dict[field_name] = coerced_value
        # END CITATION

    def __var_def(self, var_ast):
        var_name = var_ast.get("name")
        var_type = var_ast.get("var_type")
        if var_type is None:
            super().error(ErrorType.TYPE_ERROR, f"No type specified for variable '{var_name}'")
        if not self.__is_valid_type(var_type):
            super().error(ErrorType.TYPE_ERROR, f"Invalid type for variable '{var_name}'")
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
            if '.' in var_name:
                return self.__eval_field_access(var_name)
            var_def = self.env.get(var_name)
            if var_def is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            return var_def["value"]
        if expr_ast.elem_type == InterpreterBase.FCALL_NODE:
            result = self.__call_func(expr_ast)
            if result.type() == Type.VOID:
                super().error(ErrorType.TYPE_ERROR, "Cannot use void function result in an expression")
            return result
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type == InterpreterBase.NEG_NODE:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -x)
        if expr_ast.elem_type == InterpreterBase.NOT_NODE:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)
        if expr_ast.elem_type == InterpreterBase.NEW_NODE:
            return self.__eval_new(expr_ast)
        super().error(ErrorType.NAME_ERROR, f"Unknown expression node type '{expr_ast.elem_type}'")
    
    # eval new struct
    def __eval_new(self, new_ast):
        struct_type = new_ast.get("var_type")
        if struct_type not in self.struct_defs:
            super().error(ErrorType.TYPE_ERROR, f"Invalid struct type for new expression")
        field_defs = self.struct_defs[struct_type]
        initial_fields = {}
        for f_name, f_type in field_defs.items():
            initial_fields[f_name] = default_value_for_type(f_type)
        return Value(struct_type, initial_fields)

    def __eval_field_access(self, field_path):
        path_parts = field_path.split('.')
        base_var_name = path_parts[0]
        var_def = self.env.get(base_var_name)
        if var_def is None:
            super().error(ErrorType.NAME_ERROR, f"Variable '{base_var_name}' not found")
        base_value = var_def["value"]
        if base_value.type() == Type.NIL:
            super().error(ErrorType.FAULT_ERROR, f"Null reference")
        curr_val = base_value
        for field_name in path_parts[1:]:
            if curr_val.type() == Type.NIL:
                super().error(ErrorType.FAULT_ERROR, f"Null reference")
            struct_type = curr_val.type()
            if struct_type not in self.struct_defs:
                super().error(ErrorType.TYPE_ERROR, f"Cannot access field '{field_name}' of non-struct variable")
            field_def = self.struct_defs[struct_type]
            if field_name not in field_def:
                super().error(ErrorType.NAME_ERROR, f"Field not found in struct '{struct_type}'")
            fields_dict = curr_val.value()
            curr_val = fields_dict[field_name]  # Get Value object
        return curr_val  # Return Value object

    def __eval_op(self, arith_ast):
        op = arith_ast.elem_type
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))
        if op in {"+", "-", "*", "/", "<", "<=", ">", ">="}:
            if op == "+" and (left_value_obj.type() == Type.STRING or right_value_obj.type() == Type.STRING):
                # support string concat
                if left_value_obj.type() != Type.STRING or right_value_obj.type() != Type.STRING:
                    super().error(
                        ErrorType.TYPE_ERROR,
                        f"Incompatible types for + operation"
                    )
            else:
                if left_value_obj.type() != Type.INT or right_value_obj.type() != Type.INT:
                    super().error(
                        ErrorType.TYPE_ERROR,
                        f"Incompatible types for {op} operation'"
                    )
        elif op in {"||", "&&"}:
            # Coerce ints to bools for logical ops
            left_value_obj = self.__coerce_type(left_value_obj, Type.BOOL)
            right_value_obj = self.__coerce_type(right_value_obj, Type.BOOL)
            if left_value_obj.type() != Type.BOOL or right_value_obj.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible types for {op} operation"
                )
        elif op in {"==", "!="}:
            # Handle struct and nil comps
            if (left_value_obj.type() in self.struct_defs or left_value_obj.type() == Type.NIL) or \
               (right_value_obj.type() in self.struct_defs or right_value_obj.type() == Type.NIL):
                return self.__eval_struct_comparison(op, left_value_obj, right_value_obj)
            # Coerce int to bool if comparing int and bool
            if left_value_obj.type() != right_value_obj.type():
                if (left_value_obj.type() == Type.INT and right_value_obj.type() == Type.BOOL):
                    left_value_obj = self.__coerce_type(left_value_obj, Type.BOOL)
                elif (left_value_obj.type() == Type.BOOL and right_value_obj.type() == Type.INT):
                    right_value_obj = self.__coerce_type(right_value_obj, Type.BOOL)
                else:
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

    # note that struct refs equal <==> ref the same object
    def __eval_struct_comparison(self, op, left_obj, right_obj):
        left_type = left_obj.type()
        right_type = right_obj.type()
        if (left_type in self.struct_defs or left_type == Type.NIL) and \
           (right_type in self.struct_defs or right_type == Type.NIL):
            if op == "==":
                return Value(Type.BOOL, left_obj.value() is right_obj.value())
            elif op == "!=":
                return Value(Type.BOOL, left_obj.value() is not right_obj.value())
            else:
                super().error(ErrorType.TYPE_ERROR, f"Incompatible operator '{op}' for struct references or nil")
        else:
            super().error(ErrorType.TYPE_ERROR, f"Incompatible operator '{op}' for struct references or nil")

    def __eval_unary(self, arith_ast, expected_type, operation):
        value_obj = self.__eval_expr(arith_ast.get("op1"))
        if arith_ast.elem_type == '!':
            # coerce if necessary
            if value_obj.type() == Type.INT:
                value_obj = self.__coerce_type(value_obj, Type.BOOL)
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
            statements = if_ast.get("statements") or []
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
        self.__run_statement(init_ast)
        while True:
            run_for_val = self.__eval_expr(cond_ast)
            if run_for_val.type() == Type.INT:
                run_for_val = self.__coerce_type(run_for_val, Type.BOOL)
            if run_for_val.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for for loop condition: expected bool or int, got '{run_for_val.type()}'"
                )
            if not run_for_val.value():
                break
            statements = for_ast.get("statements") or []
            status, return_val = self.__run_statements(statements)
            if status == ExecStatus.RETURN:
                return status, return_val
            self.__run_statement(update_ast)
        return ExecStatus.CONTINUE, Interpreter.NIL_VALUE

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return ExecStatus.RETURN, None
        value_obj = self.__eval_expr(expr_ast)
        return ExecStatus.RETURN, value_obj

    def __is_valid_type(self, t):
        return (t in {Type.INT, Type.BOOL, Type.STRING, InterpreterBase.VOID_DEF} or
                t in self.struct_defs)

    def __check_type_compatibility(self, target_type, source_type):
        if target_type == source_type:
            return True
        # Coercion from int to bool
        if target_type == Type.BOOL and source_type == Type.INT:
            return True
        # Coercion from struct to nil and nil to struct
        if target_type in self.struct_defs and source_type == Type.NIL:
            return True
        if source_type in self.struct_defs and target_type == Type.NIL:
            return True
        return False

    def __coerce_type(self, value_obj, target_type):
        if value_obj.type() == target_type:
            return value_obj
        if value_obj.type() == Type.INT and target_type == Type.BOOL:
            coerced_val = (value_obj.value() != 0)
            return Value(Type.BOOL, coerced_val)
        return value_obj
