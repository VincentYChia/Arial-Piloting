from typing import List, Tuple, Union
import re, queue
from enum import Enum
import time
from typing import Optional
from threading import Thread, Event
from queue import Queue
from openai import ChatCompletion, Stream
from .skillset import SkillSet
from .utils import split_args, print_t


def print_debug(*args):
    print(*args)
    # pass


MiniSpecValueType = Union[int, float, bool, str, None]


def evaluate_value(value: str) -> MiniSpecValueType:
    try:
        if value.isdigit():
            return int(value)
        elif value.replace('.', '', 1).isdigit():
            return float(value)
        elif value == 'True':
            return True
        elif value == 'False':
            return False
        elif value == 'None' or len(value) == 0:
            return None
        else:
            return value.strip('\'"')
    except Exception:
        return value  # Return as-is if evaluation fails


def is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid identifier (function/variable name)"""
    if not name or not name.replace('_', '').replace('-', '').isalnum():
        return False
    return name[0].isalpha() or name[0] == '_'


class MiniSpecReturnValue:
    def __init__(self, value: MiniSpecValueType, replan: bool):
        self.value = value
        self.replan = replan

    def from_tuple(t: Tuple[MiniSpecValueType, bool]):
        return MiniSpecReturnValue(t[0], t[1])

    def default():
        return MiniSpecReturnValue(None, False)

    def error(error_msg: str, should_replan: bool = False):
        """Create an error return value with optional replanning"""
        return MiniSpecReturnValue(f"Error: {error_msg}", should_replan)

    def __repr__(self) -> str:
        return f'value={self.value}, replan={self.replan}'


class ParsingState(Enum):
    CODE = 0
    ARGUMENTS = 1
    CONDITION = 2
    LOOP_COUNT = 3
    SUB_STATEMENTS = 4


class MiniSpecProgram:
    def __init__(self, env: Optional[dict] = None, mq: queue.Queue = None, cancellation_event: Event = None) -> None:
        self.statements: List[Statement] = []
        self.depth = 0
        self.finished = False
        self.ret = False
        if env is None:
            self.env = {}
        else:
            self.env = env
        self.current_statement = Statement(self.env, cancellation_event)
        self.mq = mq
        self.cancellation_event = cancellation_event or Event()

    def parse(self, code_instance: Stream[ChatCompletion.ChatCompletionChunk] | List[str], exec: bool = False) -> bool:
        for chunk in code_instance:
            if self.cancellation_event.is_set():
                return True

            if isinstance(chunk, str):
                code = chunk
            else:
                code = chunk.choices[0].delta.content
            if code == None or len(code) == 0:
                continue
            if self.mq:
                self.mq.put(code + '\\\\')

            try:
                for c in code:
                    try:
                        if self.current_statement.parse(c, exec):
                            if len(self.current_statement.action) > 0:
                                print_debug("Adding statement: ", self.current_statement, exec)
                                self.statements.append(self.current_statement)
                            self.current_statement = Statement(self.env, self.cancellation_event)
                        if c == '{':
                            self.depth += 1
                        elif c == '}':
                            if self.depth == 0:
                                self.finished = True
                                return True
                            self.depth -= 1
                    except Exception as e:
                        print_debug(f"Parse error for character '{c}': {e}, skipping...")
                        continue
            except Exception as e:
                print_debug(f"Chunk parsing error: {e}, continuing with next chunk...")
                continue
        return False

    def eval(self) -> MiniSpecReturnValue:
        print_debug(f'Eval program: {self}, finished: {self.finished}')
        ret_val = MiniSpecReturnValue.default()
        count = 0
        while not self.finished:
            if self.cancellation_event.is_set():
                return MiniSpecReturnValue("Task cancelled", False)

            if len(self.statements) <= count:
                time.sleep(0.1)
                continue
            ret_val = self.statements[count].eval()
            if ret_val.replan or self.statements[count].ret:
                print_debug(f'RET from {self.statements[count]} with {ret_val} {self.statements[count].ret}')
                self.ret = True
                return ret_val
            count += 1
        if count < len(self.statements):
            for i in range(count, len(self.statements)):
                if self.cancellation_event.is_set():
                    return MiniSpecReturnValue("Task cancelled", False)

                ret_val = self.statements[i].eval()
                if ret_val.replan or self.statements[i].ret:
                    print_debug(f'RET from {self.statements[i]} with {ret_val} {self.statements[i].ret}')
                    self.ret = True
                    return ret_val
        return ret_val

    def __repr__(self) -> str:
        s = ''
        for statement in self.statements:
            s += f'{statement}; '
        return s


class Statement:
    execution_queue: Queue['Statement'] = None
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    cancellation_event: Event = None

    def __init__(self, env: dict, cancellation_event: Event = None) -> None:
        self.code_buffer: str = ''
        self.parsing_state: ParsingState = ParsingState.CODE
        self.condition: Optional[str] = None
        self.loop_count: Optional[int] = None
        self.action: str = ''
        self.allow_digit: bool = False
        self.executable: bool = False
        self.ret: bool = False
        self.sub_statements: Optional[MiniSpecProgram] = None
        self.env = env
        self.read_argument: bool = False
        self.cancellation_event = cancellation_event or Event()
        self.is_malformed: bool = False

    def get_env_value(self, var) -> MiniSpecValueType:
        if var not in self.env:
            print_debug(f'Warning: Variable {var} is not defined, returning None')
            return None
        return self.env[var]

    def parse(self, code: str, exec: bool = False) -> bool:
        for c in code:
            if self.cancellation_event.is_set():
                return True

            try:
                match self.parsing_state:
                    case ParsingState.CODE:
                        if c == '?' and not self.read_argument:
                            self.action = 'if'
                            self.parsing_state = ParsingState.CONDITION
                        elif c == ';' or c == '}' or c == ')':
                            if c == ')':
                                self.code_buffer += c
                                self.read_argument = False
                            self.action = self.code_buffer.strip()
                            print_debug(f'SP Action: {self.code_buffer}')

                            # Validate the action before marking as executable
                            if self.action and not self._is_valid_action(self.action):
                                print_debug(f'Warning: Invalid action "{self.action}", marking as malformed')
                                self.is_malformed = True

                            self.executable = True
                            if exec and self.action != '':
                                self.execution_queue.put(self)
                            return True
                        else:
                            if c == '(':
                                self.read_argument = True
                            if c.isalpha() or c == '_':
                                self.allow_digit = True
                            # Filter out obviously invalid characters in identifiers
                            if c.isprintable() and (c.isalnum() or c in '()_=+-*/<>!&|. \'"'):
                                self.code_buffer += c
                            else:
                                print_debug(f'Skipping invalid character: {repr(c)}')
                        if c.isdigit() and not self.allow_digit:
                            self.action = 'loop'
                            self.parsing_state = ParsingState.LOOP_COUNT
                    case ParsingState.CONDITION:
                        if c == '{':
                            print_debug(f'SP Condition: {self.code_buffer}')
                            self.condition = self.code_buffer.strip()
                            if not self._is_valid_condition(self.condition):
                                print_debug(f'Warning: Invalid condition "{self.condition}", marking as malformed')
                                self.is_malformed = True
                            self.executable = True
                            if exec:
                                self.execution_queue.put(self)
                            self.sub_statements = MiniSpecProgram(self.env, cancellation_event=self.cancellation_event)
                            self.parsing_state = ParsingState.SUB_STATEMENTS
                        else:
                            if c.isprintable():
                                self.code_buffer += c
                    case ParsingState.LOOP_COUNT:
                        if c == '{':
                            print_debug(f'SP Loop: {self.code_buffer}')
                            try:
                                self.loop_count = int(self.code_buffer.strip())
                                if self.loop_count < 0:
                                    print_debug(f'Warning: Negative loop count, setting to 1')
                                    self.loop_count = 1
                            except ValueError:
                                print_debug(f'Warning: Invalid loop count "{self.code_buffer}", defaulting to 1')
                                self.loop_count = 1
                                self.is_malformed = True
                            self.executable = True
                            if exec:
                                self.execution_queue.put(self)
                            self.sub_statements = MiniSpecProgram(self.env, cancellation_event=self.cancellation_event)
                            self.parsing_state = ParsingState.SUB_STATEMENTS
                        else:
                            if c.isdigit():
                                self.code_buffer += c
                            else:
                                print_debug(f'Invalid character in loop count: {repr(c)}, ignoring')
                    case ParsingState.SUB_STATEMENTS:
                        if self.sub_statements.parse([c]):
                            return True
            except Exception as e:
                print_debug(f'Parse error in state {self.parsing_state}: {e}')
                self.is_malformed = True
                continue
        return False

    def _is_valid_action(self, action: str) -> bool:
        """Validate if an action string is well-formed"""
        if not action:
            return False

        # Check for basic patterns: variable assignment, function call, or simple expression
        action = action.strip()

        # Variable assignment pattern
        if '=' in action:
            parts = action.split('=', 1)
            if len(parts) == 2:
                var_name = parts[0].strip()
                return is_valid_identifier(var_name) or var_name.startswith('_')

        # Function call pattern
        if '(' in action and action.endswith(')'):
            func_name = action.split('(')[0].strip()
            return is_valid_identifier(func_name)

        # Simple identifier or return statement
        if action.startswith('->'):
            return True

        return is_valid_identifier(action) or action.startswith('_') or action.replace('.', '').replace('-',
                                                                                                        '').isalnum()

    def _is_valid_condition(self, condition: str) -> bool:
        """Validate if a condition string is well-formed"""
        if not condition:
            return False

        # Check for basic comparison operators
        operators = ['>=', '<=', '>', '<', '==', '!=', '&', '|']
        has_operator = any(op in condition for op in operators)

        # If no operators, should be a simple boolean expression
        if not has_operator:
            return bool(condition.strip())

        return True  # Let the evaluation handle complex validation

    def eval(self) -> MiniSpecReturnValue:
        print_debug(f'Statement eval: {self} {self.action} {self.condition} {self.loop_count}')
        while not self.executable:
            if self.cancellation_event.is_set():
                return MiniSpecReturnValue("Task cancelled", False)
            time.sleep(0.1)

        # Handle malformed statements
        if self.is_malformed:
            print_debug(f'Skipping malformed statement: {self}')
            return MiniSpecReturnValue.error("Malformed statement", False)

        try:
            if self.action == 'if':
                ret_val = self.eval_condition(self.condition)
                if ret_val.replan:
                    return ret_val
                if isinstance(ret_val.value, str) and ret_val.value.startswith("Error:"):
                    return ret_val  # Return error without evaluating sub-statements
                if ret_val.value:
                    print_debug(f'-> eval condition statement: {self.sub_statements}')
                    ret_val = self.sub_statements.eval()
                    if ret_val.replan or self.sub_statements.ret:
                        self.ret = True
                    return ret_val
                else:
                    return MiniSpecReturnValue.default()
            elif self.action == 'loop':
                print_debug(f'-> eval loop statement: {self.loop_count} {self.sub_statements}')
                ret_val = MiniSpecReturnValue.default()
                for i in range(self.loop_count):
                    if self.cancellation_event.is_set():
                        return MiniSpecReturnValue("Task cancelled", False)

                    print_debug(f'-> loop iteration: {ret_val}')
                    ret_val = self.sub_statements.eval()
                    if ret_val.replan or self.sub_statements.ret:
                        self.ret = True
                        return ret_val
                return ret_val
            else:
                self.ret = False
                return self.eval_expr(self.action)
        except Exception as e:
            error_msg = f"Statement execution error: {str(e)}"
            print_debug(f'Error in statement eval: {error_msg}')
            return MiniSpecReturnValue.error(error_msg, False)

    def eval_function(self, func: str) -> MiniSpecReturnValue:
        print_debug(f'Eval function: {func}')

        if self.cancellation_event.is_set():
            return MiniSpecReturnValue("Task cancelled", False)

        try:
            # Basic validation
            if not func or '(' not in func:
                return MiniSpecReturnValue.error(f"Malformed function call: {func}")

            # Parse function call
            try:
                func_split = func.split('(', 1)
                name = func_split[0].strip()

                if not is_valid_identifier(name):
                    return MiniSpecReturnValue.error(f"Invalid function name: {name}")

                if len(func_split) == 2:
                    args_str = func_split[1].strip()
                    if not args_str.endswith(')'):
                        return MiniSpecReturnValue.error(f"Unclosed function call: {func}")
                    args_str = args_str[:-1]  # Remove closing parenthesis
                    args = split_args(args_str) if args_str else []
                    for i in range(len(args)):
                        args[i] = args[i].strip().strip('\'"')
                        if args[i].startswith('_'):
                            args[i] = self.get_env_value(args[i])
                else:
                    args = []
            except Exception as e:
                return MiniSpecReturnValue.error(f"Function parsing error: {str(e)}")

            # Built-in type conversions with error handling
            if name == 'int':
                try:
                    if not args:
                        return MiniSpecReturnValue.error("int() requires an argument")
                    return MiniSpecReturnValue(int(float(str(args[0]))), False)
                except (ValueError, TypeError) as e:
                    return MiniSpecReturnValue.error(f"Cannot convert '{args[0]}' to int")
            elif name == 'float':
                try:
                    if not args:
                        return MiniSpecReturnValue.error("float() requires an argument")
                    return MiniSpecReturnValue(float(args[0]), False)
                except (ValueError, TypeError) as e:
                    return MiniSpecReturnValue.error(f"Cannot convert '{args[0]}' to float")
            elif name == 'str':
                try:
                    if not args:
                        return MiniSpecReturnValue("", False)
                    return MiniSpecReturnValue(str(args[0]), False)
                except Exception:
                    return MiniSpecReturnValue.error("String conversion failed")
            else:
                # Try low-level skills first
                skill_instance = Statement.low_level_skillset.get_skill(name)
                if skill_instance is not None:
                    print_debug(f'Executing low-level skill: {skill_instance.get_name()} {args}')
                    try:
                        result = skill_instance.execute(args)
                        return MiniSpecReturnValue.from_tuple(result)
                    except Exception as e:
                        error_msg = f"Low-level skill '{name}' failed: {str(e)}"
                        print_debug(error_msg)
                        return MiniSpecReturnValue.error(error_msg)

                # Try high-level skills
                skill_instance = Statement.high_level_skillset.get_skill(name)
                if skill_instance is not None:
                    print_debug(f'Executing high-level skill: {skill_instance.get_name()}', args)
                    try:
                        result = skill_instance.execute(args)
                        print_debug(f'High-level skill result: {result}')
                        interpreter = MiniSpecProgram(cancellation_event=self.cancellation_event)
                        interpreter.parse([result])
                        interpreter.finished = True
                        val = interpreter.eval()
                        if val.value == 'rp':
                            return MiniSpecReturnValue(f'High-level skill {skill_instance.get_name()} failed', True)
                        return val
                    except Exception as e:
                        error_msg = f"High-level skill '{name}' failed: {str(e)}"
                        print_debug(error_msg)
                        return MiniSpecReturnValue.error(error_msg)

                # Skill not found
                error_msg = f'Skill {name} is not defined'
                print_debug(error_msg)
                return MiniSpecReturnValue.error(error_msg)

        except Exception as e:
            error_msg = f"Function evaluation error: {str(e)}"
            print_debug(f'Error in eval_function: {error_msg}')
            return MiniSpecReturnValue.error(error_msg)

    def eval_expr(self, var: str) -> MiniSpecReturnValue:
        print_t(f'Eval expr: {var}')

        if self.cancellation_event.is_set():
            return MiniSpecReturnValue("Task cancelled", False)

        try:
            var = var.strip()
            if not var:
                return MiniSpecReturnValue(None, False)

            if var.startswith('->'):
                self.ret = True
                return MiniSpecReturnValue(self.eval_expr(var.lstrip('->')).value, True)

            if '=' in var:
                parts = var.split('=', 1)  # Only split on first '=' to handle expressions like "x = y == z"
                if len(parts) != 2:
                    return MiniSpecReturnValue.error(f"Invalid assignment: {var}")
                var_name, expr = parts
                var_name = var_name.strip()
                expr = expr.strip()

                if not (is_valid_identifier(var_name) or var_name.startswith('_')):
                    return MiniSpecReturnValue.error(f"Invalid variable name: {var_name}")

                print_t(f'Eval expr var assign: {var_name} {expr}')
                ret_val = self.eval_expr(expr)
                if not ret_val.replan and not (isinstance(ret_val.value, str) and ret_val.value.startswith("Error:")):
                    self.env[var_name] = ret_val.value
                return ret_val

            # Handle arithmetic operators
            arithmetic_ops = ['+', '-', '*', '/']
            for op in arithmetic_ops:
                if op in var:
                    try:
                        operands = var.split(op)
                        if len(operands) < 2:
                            continue

                        if op == '+':
                            val = 0
                            for operand in operands:
                                operand_val = self.eval_expr(operand.strip()).value
                                if operand_val is None:
                                    operand_val = 0
                                val += float(operand_val)
                        elif op == '-':
                            val = float(self.eval_expr(operands[0].strip()).value or 0)
                            for operand in operands[1:]:
                                operand_val = self.eval_expr(operand.strip()).value
                                if operand_val is None:
                                    operand_val = 0
                                val -= float(operand_val)
                        elif op == '*':
                            val = 1
                            for operand in operands:
                                operand_val = self.eval_expr(operand.strip()).value
                                if operand_val is None:
                                    operand_val = 0
                                val *= float(operand_val)
                        elif op == '/':
                            val = float(self.eval_expr(operands[0].strip()).value or 0)
                            for operand in operands[1:]:
                                operand_val = self.eval_expr(operand.strip()).value
                                if operand_val is None or float(operand_val) == 0:
                                    return MiniSpecReturnValue.error("Division by zero or None")
                                val /= float(operand_val)
                        return MiniSpecReturnValue(val, False)
                    except (ValueError, TypeError) as e:
                        return MiniSpecReturnValue.error(f"Arithmetic operation failed: {str(e)}")

            if var.startswith('_'):
                return MiniSpecReturnValue(self.get_env_value(var), False)
            elif var in ['True', 'False', 'None']:
                return MiniSpecReturnValue(evaluate_value(var), False)
            elif var[0].isalpha() or (len(var) > 0 and var[0] == '_'):
                return self.eval_function(var)
            else:
                return MiniSpecReturnValue(evaluate_value(var), False)

        except Exception as e:
            error_msg = f"Expression evaluation error: {str(e)}"
            print_debug(f'Error in eval_expr: {error_msg}')
            return MiniSpecReturnValue.error(error_msg)

    def eval_condition(self, condition: str) -> MiniSpecReturnValue:
        if self.cancellation_event.is_set():
            return MiniSpecReturnValue("Task cancelled", False)

        try:
            condition = condition.strip()
            if not condition:
                return MiniSpecReturnValue.error("Empty condition")

            if '&' in condition:
                conditions = condition.split('&')
                cond = True
                for c in conditions:
                    ret_val = self.eval_condition(c.strip())
                    if ret_val.replan:
                        return ret_val
                    # Handle error cases gracefully
                    if isinstance(ret_val.value, str) and ret_val.value.startswith("Error:"):
                        return ret_val
                    cond = cond and bool(ret_val.value)
                return MiniSpecReturnValue(cond, False)
            if '|' in condition:
                conditions = condition.split('|')
                for c in conditions:
                    ret_val = self.eval_condition(c.strip())
                    if ret_val.replan:
                        return ret_val
                    # Handle error cases gracefully
                    if isinstance(ret_val.value, str) and ret_val.value.startswith("Error:"):
                        continue  # Try next condition
                    if bool(ret_val.value):
                        return MiniSpecReturnValue(True, False)
                return MiniSpecReturnValue(False, False)

            # Parse comparison operators (order matters - check >= before >)
            operators = ['>=', '<=', '==', '!=', '>', '<']
            match = None
            comparator = None
            for op in operators:
                if op in condition:
                    match = op
                    comparator = op
                    break

            if not match:
                # No comparison operator, evaluate as boolean expression
                ret_val = self.eval_expr(condition)
                if isinstance(ret_val.value, str) and ret_val.value.startswith("Error:"):
                    return ret_val
                return MiniSpecReturnValue(bool(ret_val.value), False)

            parts = condition.split(comparator, 1)
            if len(parts) != 2:
                return MiniSpecReturnValue.error(f"Invalid condition format: {condition}")

            operand_1_str, operand_2_str = parts
            operand_1 = self.eval_expr(operand_1_str.strip())
            if operand_1.replan:
                return operand_1
            operand_2 = self.eval_expr(operand_2_str.strip())
            if operand_2.replan:
                return operand_2

            # Handle error cases
            if (isinstance(operand_1.value, str) and operand_1.value.startswith("Error:")) or \
                    (isinstance(operand_2.value, str) and operand_2.value.startswith("Error:")):
                return MiniSpecReturnValue.error("Cannot evaluate condition due to operand errors")

            # Convert string representations of booleans
            for operand in [operand_1, operand_2]:
                if isinstance(operand.value, str):
                    if operand.value == 'True':
                        operand.value = True
                    elif operand.value == 'False':
                        operand.value = False

            print_debug(f'Condition ops: {operand_1.value} {comparator} {operand_2.value}')

            # Handle None values
            if operand_1.value is None or operand_2.value is None:
                if comparator in ['!=', '==']:
                    return MiniSpecReturnValue(operand_1.value != operand_2.value if comparator == '!='
                                               else operand_1.value == operand_2.value, False)
                else:
                    return MiniSpecReturnValue.error(f"Cannot compare None with {comparator}")

            # Type conversion for numeric comparisons
            if (type(operand_1.value) in [int, float] and type(operand_2.value) in [int, float]):
                operand_1.value = float(operand_1.value)
                operand_2.value = float(operand_2.value)
            elif type(operand_1.value) != type(operand_2.value):
                if comparator == '!=':
                    return MiniSpecReturnValue(True, False)
                elif comparator == '==':
                    return MiniSpecReturnValue(False, False)
                else:
                    return MiniSpecReturnValue.error(
                        f'Cannot compare different types: {type(operand_1.value).__name__} {comparator} {type(operand_2.value).__name__}')

            # Perform comparison
            try:
                if comparator == '>':
                    cmp = operand_1.value > operand_2.value
                elif comparator == '<':
                    cmp = operand_1.value < operand_2.value
                elif comparator == '>=':
                    cmp = operand_1.value >= operand_2.value
                elif comparator == '<=':
                    cmp = operand_1.value <= operand_2.value
                elif comparator == '==':
                    cmp = operand_1.value == operand_2.value
                elif comparator == '!=':
                    cmp = operand_1.value != operand_2.value
                else:
                    return MiniSpecReturnValue.error(f'Invalid comparator: {comparator}')

                return MiniSpecReturnValue(cmp, False)
            except TypeError as e:
                return MiniSpecReturnValue.error(f"Comparison failed: {str(e)}")

        except Exception as e:
            error_msg = f"Condition evaluation error: {str(e)}"
            print_debug(f'Error in eval_condition: {error_msg}')
            return MiniSpecReturnValue.error(error_msg)

    def __repr__(self) -> str:
        s = ''
        if self.action == 'if':
            s += f'if {self.condition}'
        elif self.action == 'loop':
            s += f'[{self.loop_count}]'
        else:
            s += f'{self.action}'
        if self.sub_statements is not None:
            s += ' {'
            for statement in self.sub_statements.statements:
                s += f'{statement}; '
            s += '}'
        return s


_queue = Queue()
self.message_queue = message_queue


def cancel(self):
    """Cancel the current execution"""
    print_debug("[CANCEL] Setting cancellation event")
    self.cancellation_event.set()

    while not Statement.execution_queue.empty():
        try:
            Statement.execution_queue.get_nowait()
        except queue.Empty:
            break

    self.ret_queue.put(MiniSpecReturnValue("Task cancelled", False))


def _fix_scan_goto_patterns(self, code_text: str) -> str:
    """
    Simple pattern fixer for common scan+goto anti-patterns
    Fixes: ?sa('obj'){g('obj')} → _tmp=sa('obj');?_tmp!=False{g(_tmp)}
    """
    import re

    # Pattern: ?function('param'){g('param')} or ?function('param'){goto('param')}
    scan_functions = ['sa', 'yc', 'ye', 's']  # scan_abstract, yolo_classify, yoloe, scan
    goto_functions = ['g', 'goto']

    original_code = code_text
    fixes_applied = []

    for scan_func in scan_functions:
        for goto_func in goto_functions:
            # Pattern: ?scan_func('param'){goto_func('param')}
            pattern = rf'\?{scan_func}\(([\'"])([^\'\"]+)\1\)\s*\{{\s*{goto_func}\(\1\2\1\)\s*\}}'

            def replace_pattern(match):
                quote_char = match.group(1)  # ' or "
                param = match.group(2)  # the parameter

                # Generate unique temp variable name
                temp_var = f'_tmp{len(fixes_applied)}'

                # Create fixed pattern: _tmp=sa('param');?_tmp!=False{g(_tmp)}
                fixed = f'{temp_var}={scan_func}({quote_char}{param}{quote_char});?{temp_var}!=False{{{goto_func}({temp_var})}}'

                fixes_applied.append(f"{scan_func}('{param}')→{goto_func}() pattern")
                return fixed

            code_text = re.sub(pattern, replace_pattern, code_text)

    if fixes_applied:
        print_debug(f"[PATTERN FIX] Applied {len(fixes_applied)} scan+goto pattern fixes:")
        for fix in fixes_applied:
            print_debug(f"  - Fixed: {fix}")
        print_debug(f"[PATTERN FIX] Before: {original_code}")
        print_debug(f"[PATTERN FIX] After:  {code_text}")

    return code_text


def execute(self, code: Stream[ChatCompletion.ChatCompletionChunk] | List[str]) -> MiniSpecReturnValue:
    print_t(f'>>> Get a stream')

    self.cancellation_event.clear()

    # Convert stream/list to string for pattern fixing
    if isinstance(code, list):
        code_text = ''.join(code)
    else:
        # For streams, we need to collect all chunks first
        code_chunks = []
        for chunk in code:
            if isinstance(chunk, str):
                code_chunks.append(chunk)
            else:
                content = chunk.choices[0].delta.content
                if content:
                    code_chunks.append(content)
        code_text = ''.join(code_chunks)

    # Apply pattern fixes
    fixed_code_text = self._fix_scan_goto_patterns(code_text)

    # Convert back to list format for parsing
    fixed_code = [fixed_code_text]

    self.execution_history = []
    self.timestamp_get_plan = time.time()
    program = MiniSpecProgram(mq=self.message_queue, cancellation_event=self.cancellation_event)
    program.parse(fixed_code, True)
    self.program_count = len(program.statements)
    t2 = time.time()
    print_t(">>> Program: ", program, "Time: ", t2 - self.timestamp_get_plan)


def executor(self):
    while True:
        if not Statement.execution_queue.empty():
            if self.timestamp_start_execution is None:
                self.timestamp_start_execution = time.time()
                print_t(">>> Start execution")

            try:
                statement = Statement.execution_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            print_debug(f'Queue get statement: {statement}')

            if self.cancellation_event.is_set():
                self.ret_queue.put(MiniSpecReturnValue("Task cancelled", False))
                return

            try:
                ret_val = statement.eval()
                print_t(f'Queue statement done: {statement}')

                # Check if this was an error and we're at the last statement
                is_error = isinstance(ret_val.value, str) and ret_val.value.startswith("Error:")
                is_last_statement = self.program_count == 1

                if is_error and is_last_statement:
                    print_debug(f'Last statement failed with error: {ret_val.value}, triggering replan')
                    # Clear remaining queue and trigger replan
                    while not Statement.execution_queue.empty():
                        try:
                            Statement.execution_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.ret_queue.put(MiniSpecReturnValue(ret_val.value, True))  # Trigger replan
                    return
                elif is_error:
                    print_debug(f'Statement failed with error: {ret_val.value}, continuing...')
                    # Continue with execution for non-last statements
                elif statement.ret:
                    while not Statement.execution_queue.empty():
                        try:
                            Statement.execution_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.ret_queue.put(ret_val)
                    return

            except Exception as e:
                # This is a last-resort catch for any unhandled exceptions
                error_msg = f"Unhandled error in statement execution: {str(e)}"
                print_debug(error_msg)
                ret_val = MiniSpecReturnValue.error(error_msg)

                # If this is the last statement, trigger replan
                if self.program_count == 1:
                    print_debug(f'Last statement crashed with unhandled error, triggering replan')
                    while not Statement.execution_queue.empty():
                        try:
                            Statement.execution_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.ret_queue.put(MiniSpecReturnValue(error_msg, True))  # Trigger replan
                    return

            self.execution_history.append(statement)
            self.program_count -= 1

            if self.program_count == 0:
                self.timestamp_end_execution = time.time()
                print_t(f'>>> Execution time: {self.timestamp_end_execution - self.timestamp_start_execution}')
                self.timestamp_start_execution = None
                self.ret_queue.put(ret_val)
                return
        else:
            time.sleep(0.005)

            if self.cancellation_event.is_set():
                return