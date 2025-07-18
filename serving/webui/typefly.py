#!/usr/bin/env python3
"""
Ollama MiniSpec Model Testing Script
Tests multiple Ollama models with MiniSpec code generation task
"""

import requests
import json
import csv
import time
from datetime import datetime
import argparse
import sys


class OllamaModelTester:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.minispec_prompt = """You are a MiniSpec code generator. Given reasoning about a robot task, generate the appropriate MiniSpec program to execute the plan.
Available Skills
High-level skills:
s (scan): Rotate to find object when not in current scene
sa (scan_abstract): Rotate to find abstract object by description when not in current scene
Low-level skills:
Movement: mf (move_forward), mb (move_backward), ml (move_left), mr (move_right), mu (move_up), md (move_down)
Rotation: tc (turn_cw), tu (turn_ccw)
Utilities: d (delay), tp (take_picture), rp (re_plan), t (time)
Vision: iv (is_visible), ox (object_x), oy (object_y), ow (object_width), oh (object_height), od (object_dis)
Logic: p (probe), l (log), g (goto)
Rules
Use high-level skills when available
Invoke skills using their short names
If a skill has no arguments, call it without parentheses
Generate response as a single MiniSpec program sentence
All statements must end with ;
Conditionals must start with ? - Format: ?condition{{statements}};
Use single quotes for string arguments: s('object')
Reasoning Input
{reasoning}

Generate the Minispec Program:
"""

        self.reasoning_input = "no bottle instance in the scene, so we use scan to find bottle"

        # Results tracking
        self.results = []
        self.seen_outputs = set()  # Track duplicates

    def test_model(self, model_name, temperature=0.0):
        """Test a single model with given temperature"""
        prompt = self.minispec_prompt.format(reasoning=self.reasoning_input)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            print(f"Testing {model_name} (temp: {temperature})...")

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=15
            )
            response.raise_for_status()

            result = response.json()
            output = result.get('response', '').strip()

            # Analyze the output
            analysis = self.analyze_output(output)

            # Record result
            result_data = {
                'model': model_name,
                'temperature': temperature,
                'timestamp': datetime.now().isoformat(),
                'output': output,
                'category': analysis['category'],
                'is_valid_minispec': analysis['is_valid_minispec'],
                'is_duplicate': analysis['is_duplicate'],
                'notes': analysis['notes']
            }

            self.results.append(result_data)
            self.seen_outputs.add(output)

            print(f"  Result: {analysis['category']}")
            return result_data

        except requests.exceptions.RequestException as e:
            print(f"  Error testing {model_name}: {e}")
            error_data = {
                'model': model_name,
                'temperature': temperature,
                'timestamp': datetime.now().isoformat(),
                'output': f"REQUEST_ERROR: {str(e)}",
                'category': 'request_error',
                'is_valid_minispec': False,
                'is_duplicate': False,
                'notes': f"Request failed: {str(e)}"
            }
            self.results.append(error_data)
            return error_data

    def validate_minispec(self, output):
        """Validate if output is valid MiniSpec syntax"""
        try:
            import re

            # Remove whitespace
            code = output.strip()

            # Check if empty
            if not code:
                return False, "Empty output"

            # Check if ends with semicolon
            if not code.endswith(';'):
                return False, "Missing semicolon"

            # Basic pattern matching for MiniSpec syntax
            # Function calls: word() or word('arg') or word("arg")
            func_pattern = r"\b[a-z_][a-z0-9_]*\s*\([^)]*\)"

            # Conditionals: ?condition{...}
            conditional_pattern = r"\?[^{]+\{[^}]*\}"

            # Simple statements: word; or word();
            simple_pattern = r"\b[a-z_][a-z0-9_]*\s*;"

            # Check for basic MiniSpec patterns
            has_function = bool(re.search(func_pattern, code, re.IGNORECASE))
            has_conditional = bool(re.search(conditional_pattern, code))
            has_simple = bool(re.search(simple_pattern, code, re.IGNORECASE))

            # Check for invalid characters or patterns
            # Allow alphanumeric, quotes, parentheses, semicolons, question marks, braces, equals, underscores
            valid_chars = re.match(r"^[a-zA-Z0-9_'\"()\{\};?=\s<>!&|]*$", code)

            if not valid_chars:
                return False, "Contains invalid characters"

            # Check balanced braces and parentheses
            brace_count = code.count('{') - code.count('}')
            paren_count = code.count('(') - code.count(')')

            if brace_count != 0:
                return False, "Unbalanced braces"
            if paren_count != 0:
                return False, "Unbalanced parentheses"

            # If we have valid patterns and structure, consider it valid
            if has_function or has_conditional or has_simple:
                return True, "Valid MiniSpec syntax"
            else:
                return False, "No valid MiniSpec patterns found"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def analyze_output(self, output):
        """Analyze model output and categorize result"""
        is_duplicate = output in self.seen_outputs

        # Validate MiniSpec syntax
        is_valid_minispec, minispec_error = self.validate_minispec(output)

        # Determine category
        if is_duplicate:
            category = 'duplicate'
            notes = 'Duplicate response'
        elif is_valid_minispec:
            category = 'valid_minispec'
            notes = 'Valid MiniSpec syntax'
        else:
            category = 'invalid_syntax'
            notes = f'Invalid syntax: {minispec_error}'

        return {
            'category': category,
            'is_valid_minispec': is_valid_minispec,
            'is_duplicate': is_duplicate,
            'notes': notes
        }

    def test_models(self, model_list, temperatures=None):
        """Test multiple models with given temperatures"""
        if temperatures is None:
            temperatures = [0.0]

        print(f"Testing {len(model_list)} models with temperatures: {temperatures}")
        print(f"Reasoning input: {self.reasoning_input}")
        print("-" * 50)

        for model in model_list:
            for temp in temperatures:
                self.test_model(model, temp)
                time.sleep(1)  # Brief pause between requests

    def save_results(self, filename=None):
        """Save results to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ollama_minispec_results_{timestamp}.csv"

        fieldnames = [
            'model', 'temperature', 'timestamp', 'output', 'category',
            'is_valid_minispec', 'is_duplicate', 'notes'
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"\nResults saved to: {filename}")
        self.print_summary()

    def print_summary(self):
        """Print summary of results"""
        if not self.results:
            return

        total = len(self.results)
        categories = {}

        for result in self.results:
            cat = result['category']
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\nSummary of {total} tests:")
        for category, count in sorted(categories.items()):
            percentage = (count / total) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")


# =============================================================================
# CONFIGURATION - Edit these settings as needed
# =============================================================================

# Models to test
MODELS = [
    'robot-pilot-writer23:latest',
    'robot-pilot-writer25:latest',
    'robot-pilot-writer26:latest',
    'robot-pilot-writer27:latest',
    'robot-pilot-writer28:latest',

]

# Temperature settings - choose ONE of these options:

# Option 1: Single temperature
# TEMPERATURES = [0.0]

# Option 2: Multiple specific temperatures
# TEMPERATURES = [0.0, 0.5, 1.0]

# Option 3: Temperature range
TEMP_START = 0.0
TEMP_END = 0.5
TEMP_STEP = 0.1

# Other settings
OLLAMA_BASE_URL = "http://localhost:11434"
OUTPUT_FILENAME = None  # None = auto-generate with timestamp


# =============================================================================

def generate_temperature_range(start, end, step):
    """Generate temperature range"""
    temps = []
    temp = start
    while temp <= end:
        temps.append(round(temp, 1))
        temp += step
    return temps


def main():
    # Use temperature range if TEMP_START is defined, otherwise use TEMPERATURES
    try:
        temperatures = generate_temperature_range(TEMP_START, TEMP_END, TEMP_STEP)
    except NameError:
        temperatures = TEMPERATURES

    # Initialize tester
    tester = OllamaModelTester(base_url=OLLAMA_BASE_URL)

    # Run tests
    try:
        tester.test_models(MODELS, temperatures)
        tester.save_results(OUTPUT_FILENAME)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        if tester.results:
            print("Saving partial results...")
            tester.save_results(OUTPUT_FILENAME)
        sys.exit(1)


if __name__ == "__main__":
    main()
