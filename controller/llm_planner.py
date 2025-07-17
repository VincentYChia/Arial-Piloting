import csv
import os
from datetime import datetime
import os, ast
from typing import Optional, Tuple
import time

from .skillset import SkillSet
from .llm_wrapper import LLMWrapper, GPT3, GPT4, TINY_LLAMA
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_t
from .minispec_interpreter import MiniSpecValueType, evaluate_value
from .abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class SimpleCsvLogger:
    def __init__(self, log_dir=None):
        # Default to Windows Downloads folder if running from WSL
        if log_dir is None:
            log_dir = self._get_windows_downloads_path()

        # Try to create directory, with fallback if permission denied
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[LOG] Using log directory: {log_dir}")
        except PermissionError:
            print(f"[LOG] Permission denied for {log_dir}, using fallback location")
            log_dir = "./Arial-Piloting Logs"
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"[LOG] Error creating directory {log_dir}: {e}, using fallback")
            log_dir = "./Arial-Piloting Logs"
            os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"flight_log_{timestamp}.csv")

        print(f"[LOG] Saving flight logs to: {os.path.abspath(self.csv_path)}")

        # Create CSV with headers
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'date', 'time_of_day', 'user_input',
                'model1_reasoning', 'model2_minispec', 'drone_result',
                'errors', 'execution_time_sec', 'reasoning_model', 'minispec_model',
                'reasoning_temperature', 'minispec_temperature'
            ])

    def _get_windows_downloads_path(self):
        """Get Windows Downloads folder path, works from WSL or Windows"""

        # Method 1: Check for ARIAL_LOG_PATH environment variable
        env_path = os.getenv('ARIAL_LOG_PATH')
        if env_path:
            return env_path

        # Method 2: Try to detect Windows username and build path
        windows_user = self._get_windows_username()
        if windows_user:
            if self._is_wsl():
                # WSL format: /mnt/c/Users/username/Downloads/Arial-Piloting Logs
                potential_path = f"/mnt/c/Users/{windows_user}/Downloads/Arial-Piloting Logs"
                # Check if we can access this path before returning it
                try:
                    parent_dir = f"/mnt/c/Users/{windows_user}/Downloads"
                    if os.path.exists(parent_dir) and os.access(parent_dir, os.W_OK):
                        return potential_path
                    else:
                        print(f"[LOG] Cannot access {parent_dir}, using fallback")
                except:
                    print(f"[LOG] Error checking access to {parent_dir}, using fallback")
            else:
                # Windows format: C:\Users\username\Downloads\Arial-Piloting Logs
                return rf"C:\Users\{windows_user}\Downloads\Arial-Piloting Logs"

        # Method 3: Fallback to local folder
        print("[LOG] Using local folder for flight logs")
        return "./Arial-Piloting Logs"

    def _get_windows_username(self):
        """Try to get Windows username from various sources"""

        # Try Windows environment variables that might be accessible from WSL
        for var in ['LOGNAME', 'USER', 'LNAME', 'USERNAME']:
            user = os.getenv(var)
            if user and user != 'root':
                return user

        # Try to extract from WSL working directory path
        # If we're in something like /home/vincentyc/TypeFly,
        # we can't directly infer Windows username, but we can try other methods

        # Try reading Windows registry through WSL (if available)
        try:
            result = os.popen('cmd.exe /c "echo %USERNAME%"').read().strip()
            if result and not result.startswith('cmd.exe'):
                return result
        except:
            pass

        # Try PowerShell command through WSL
        try:
            result = os.popen('powershell.exe -c "$env:USERNAME"').read().strip()
            if result and not result.startswith('powershell.exe'):
                return result
        except:
            pass

        return None

    def _is_wsl(self):
        """Detect if running in WSL"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False

    def log(self, user_input, model1_reasoning, model2_minispec,
            execution_time, reasoning_model, minispec_model,
            reasoning_temperature, minispec_temperature,
            drone_result="", errors=""):

        now = datetime.now()
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                user_input,
                model1_reasoning,
                model2_minispec,
                drone_result,
                errors,
                execution_time,
                reasoning_model,
                minispec_model,
                reasoning_temperature,
                minispec_temperature
            ])


class LLMPlanner():
    def __init__(self, robot_type: RobotType):
        print_t("[P] Initializing dual-model LLM planner with probe support...")

        # MODEL CONFIGURATION (EASILY MODIFIABLE)
        # Both models use local Ollama for now
        self.reasoning_model = "robot-pilot-reason9"
        self.minispec_model = "robot-pilot-writer17"

        # Temperature settings for each model
        self.reasoning_temperature = 0.5  # Slightly higher for reasoning creativity
        self.minispec_temperature = 0.0  # Very low for precise code generation

        print_t(f"[P] Reasoning model: {self.reasoning_model} (temp: {self.reasoning_temperature})")
        print_t(f"[P] MiniSpec model: {self.minispec_model} (temp: {self.minispec_temperature})")

        # Initialize two separate LLM instances
        self.reasoning_llm = LLMWrapper(temperature=self.reasoning_temperature)
        self.minispec_llm = LLMWrapper(temperature=self.minispec_temperature)

        self.robot_type = robot_type
        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'

        self.type_folder_name = type_folder_name

        # Add CSV logger - automatically detects Windows Downloads folder
        self.csv_logger = SimpleCsvLogger()

        # Initialize inline prompts
        self._init_prompts()

        print_t("[P] Dual-model planner with probe support initialized successfully")

    def _init_prompts(self):
        """Initialize all prompts as inline strings matching fine-tuning format"""
        print_t("[P] Loading inline prompts...")

        # STAGE 1: REASONING PROMPT - MATCHING FINE-TUNING FORMAT
        self.reasoning_prompt = """You are a robot pilot reasoning system. Analyze the given scene and task to provide detailed reasoning about what actions should be taken.
## Available Skills
High-level skills:
s (scan): Rotate to find object when not in current scene
sa (scan_abstract): Rotate to find abstract object by description when not in current scene
Low-level skills:
Movement: mf (move_forward), mb (move_backward), ml (move_left), mr (move_right), mu (move_up), md (move_down)
Rotation: tc (turn_cw), tu (turn_ccw)
Utilities: d (delay), tp (take_picture), rp (re_plan), t (time)
Vision: iv (is_visible), ox (object_x), oy (object_y), ow (object_width), oh (object_height), od (object_dis)
Logic: p (probe), l (log), g (goto)
## Input Format
- **Scene description:** Objects in current view with IDs, locations, and sizes (coordinates 0-1)
- **Task description:** Natural language instructions starting with "[A]" (action) or "[Q]" (question)
- **Execution history:** Previous actions taken (provided during replanning)
## Your Role
Analyze the scene and task, then provide detailed reasoning about:
1. What the task is asking for
2. What objects/information are relevant
3. What approach should be taken
4. What skills or sequence of actions would be most appropriate
## Current Input
**Scene:** {scene_description}
**Task:** {task_description}
**Execution History:** {execution_history}
Provide your reasoning:"""

        # STAGE 2: MINISPEC WRITER PROMPT - MATCHING FINE-TUNING FORMAT
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

        # PROBE PROMPT - RESTORED
        self.probe_prompt = """You are given a scene description and a question. You should output the answer to the question based on the scene description.
The scene description contains listed objects with their respective names, locations, and sizes.
The question is a string that asks about the scene or the objects in the scene.
For yes-or-no questions, output with 'True' or 'False' only.
For object identification, output the object's name (if there are multiple same objects, output the target one with x value). If the object is not in the list, output with 'False'.
For counting questions, output the exact number of target objects.
For general questions, output a brief, single-sentence answer.
Input Format:
Scene Description:[List of Objects with Attributes]
Question:[A String]
Output Format:
[A String]
Here are some examples:
Example 1:
Scene Description:[person x:0.59 y:0.55 width:0.81 height:0.91, bottle x:0.85 y:0.54 width:0.21 height:0.93]
Question:'Any drinkable target here?'
Output:bottle
Example 2:
Scene Description:[]
Question:'Any table in the room?'
Output:False
Example 3:
Scene Description:[chair x:0.1 y:0.35 width:0.56 height:0.41, chair x:0.49 y:0.59 width:0.61 height:0.35]
Question:'How many chairs you can see?'
Output:2
Example 4:
Scene Description:[bottle x:0.1 y:0.35 width:0.56 height:0.41, chair x:0.49 y:0.59 width:0.61 height:0.35]
Question:'Any edible target here?'
Output:False
Example 5:
Scene Description:[chair x:0.18 y:0.5 width:0.43 height:0.7, chair x:0.6 y:0.3 width:0.08 height:0.09, book x:0.62 y:0.26 width:0.23 height:0.17]
Question:'Any chair with a laptop on it?'
Output:chair[0.6]

Scene Description:{scene_description}
Question:{question}
Please give the content of results only, don't include 'Output:' in the results."""

        # Load additional data from files for reasoning stage
        try:
            with open(os.path.join(CURRENT_DIR, f"./assets/{self.type_folder_name}/guides.txt"), "r") as f:
                self.guides = f.read()
                print_t("[P] Loaded guides from file")
        except FileNotFoundError as e:
            print_t(f"[P] Warning: Could not load guides: {e}")
            self.guides = "Follow task instructions carefully and use appropriate skills."

    # MODEL CONFIGURATION METHODS
    def set_reasoning_model(self, model_name: str, temperature: float = None):
        """Set the model and temperature for reasoning stage"""
        self.reasoning_model = model_name
        if temperature is not None:
            self.reasoning_temperature = temperature
            self.reasoning_llm = LLMWrapper(temperature=temperature)
        print_t(f"[P] Reasoning model updated: {model_name} (temp: {self.reasoning_temperature})")

    def set_minispec_model(self, model_name: str, temperature: float = None):
        """Set the model and temperature for MiniSpec generation stage"""
        self.minispec_model = model_name
        if temperature is not None:
            self.minispec_temperature = temperature
            self.minispec_llm = LLMWrapper(temperature=temperature)
        print_t(f"[P] MiniSpec model updated: {model_name} (temp: {self.minispec_temperature})")

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        """Initialize skillsets and vision"""
        print_t("[P] Initializing skillsets and vision...")
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill
        print_t("[P] Skillsets and vision initialized")

    def plan_reasoning(self, task_description: str, scene_description: str,
                       execution_history: Optional[str] = None) -> str:
        """Stage 1: Generate reasoning for the task"""
        print_t("[P] ====== STAGE 1: REASONING ======")
        print_t(f"[P] Task: {task_description}")
        print_t(f"[P] Scene: {scene_description}")
        print_t(f"[P] History: {execution_history}")

        start_time = time.time()

        # Format the reasoning prompt - MATCHING FINE-TUNING FORMAT
        prompt = self.reasoning_prompt.format(
            scene_description=scene_description,
            task_description=task_description,
            execution_history=execution_history or "None"
        )

        print_t(f"[P] Sending reasoning request to model: {self.reasoning_model}")
        print_t(f"[P] Reasoning prompt length: {len(prompt)} characters")
        print_t(f"[P] Reasoning prompt preview: {prompt[:200]}...")

        # Make the LLM call
        response = self.reasoning_llm.request(prompt, self.reasoning_model, stream=False)

        reasoning_time = time.time() - start_time
        print_t(f"[P] Reasoning completed in {reasoning_time:.2f}s")
        print_t(f"[P] Raw reasoning response: {response}")

        # Extract reasoning (remove "Reason:" prefix if present)
        if response.startswith("Reason:"):
            reasoning = response[7:].strip()
        else:
            reasoning = response.strip()

        print_t(f"[P] Extracted reasoning: {reasoning}")
        return reasoning

    def plan_minispec(self, reasoning: str) -> str:
        """Stage 2: Generate MiniSpec code based ONLY on reasoning"""
        print_t("[P] ====== STAGE 2: MINISPEC GENERATION ======")
        print_t(f"[P] Input reasoning: {reasoning}")

        start_time = time.time()

        # Format the MiniSpec prompt - MATCHING FINE-TUNING FORMAT
        prompt = self.minispec_prompt.format(reasoning=reasoning)

        print_t(f"[P] Sending MiniSpec request to model: {self.minispec_model}")
        print_t(f"[P] MiniSpec prompt length: {len(prompt)} characters")
        print_t(f"[P] MiniSpec prompt preview: {prompt[:200]}...")

        # Make the LLM call
        response = self.minispec_llm.request(prompt, self.minispec_model, stream=False)

        minispec_time = time.time() - start_time
        print_t(f"[P] MiniSpec generation completed in {minispec_time:.2f}s")
        print_t(f"[P] Raw MiniSpec response: {response}")

        # Extract code (remove "Response:" prefix if present)
        if response.startswith("Response:"):
            code = response[9:].strip()
        else:
            code = response.strip()

        # Remove trailing semicolon if present
        if code.endswith(';'):
            code = code[:-1]

        print_t(f"[P] Extracted MiniSpec code: {code}")
        return code

    def plan(self, task_description: str, scene_description: Optional[str] = None, error_message: Optional[str] = None,
             execution_history: Optional[str] = None) -> Tuple[str, str]:
        """
        Main planning method - orchestrates reasoning and MiniSpec generation

        Returns:
            Tuple[str, str]: (reasoning, minispec_code)
        """
        print_t("[P] ========== DUAL-STAGE PLANNING START ==========")
        total_start_time = time.time()

        # Prepare task description
        if not task_description.startswith("["):
            task_description = "[A] " + task_description
        print_t(f"[P] Formatted task: {task_description}")

        # Get scene description if not provided
        if scene_description is None:
            scene_description = self.vision_skill.get_obj_list()
            print_t(f"[P] Retrieved scene description: {scene_description}")

        try:
            # Stage 1: Generate reasoning (gets full context)
            reasoning = self.plan_reasoning(task_description, scene_description, execution_history)

            # Stage 2: Generate MiniSpec code (gets ONLY reasoning)
            code = self.plan_minispec(reasoning)

            total_time = time.time() - total_start_time
            print_t(f"[P] ========== DUAL-STAGE PLANNING COMPLETE in {total_time:.2f}s ==========")
            print_t(f"[P] Final reasoning: {reasoning}")
            print_t(f"[P] Final MiniSpec code: {code}")

            # Log successful operation
            self.csv_logger.log(
                user_input=task_description,
                model1_reasoning=reasoning,
                model2_minispec=code,
                execution_time=total_time,
                reasoning_model=self.reasoning_model,
                minispec_model=self.minispec_model,
                reasoning_temperature=self.reasoning_temperature,
                minispec_temperature=self.minispec_temperature,
                errors=error_message or ""
            )

            return reasoning, code

        except Exception as e:
            print_t(f"[P] ERROR in dual-stage planning: {e}")
            print_t(f"[P] Returning fallback response")

            # Log failed operation
            self.csv_logger.log(
                user_input=task_description,
                model1_reasoning=f"ERROR: {e}",
                model2_minispec="log('Planning failed')",
                execution_time=time.time() - total_start_time,
                reasoning_model=self.reasoning_model,
                minispec_model=self.minispec_model,
                reasoning_temperature=self.reasoning_temperature,
                minispec_temperature=self.minispec_temperature,
                errors=str(e)
            )

            return f"Error in planning: {e}", "log('Planning failed')"

    # PROBE FUNCTIONALITY RESTORED
    def probe(self, question: str) -> Tuple[MiniSpecValueType, bool]:
        """Probe method for scene questioning using reasoning model"""
        print_t(f"[P] === PROBE REQUEST ===")
        print_t(f"[P] Question: {question}")

        prompt = self.probe_prompt.format(
            scene_description=self.vision_skill.get_obj_list(),
            question=question
        )

        print_t(f"[P] Sending probe to reasoning model: {self.reasoning_model}")
        print_t(f"[P] Probe prompt length: {len(prompt)} characters")
        response = self.reasoning_llm.request(prompt, self.reasoning_model)

        print_t(f"[P] Probe response: {response}")
        return evaluate_value(response), False