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
                'reasoning_temperature', 'minispec_temperature', 'combined_mode'
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
            drone_result="", errors="", combined_mode=False):

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
                minispec_temperature,
                combined_mode
            ])


class LLMPlanner():
    def __init__(self, robot_type: RobotType):
        print_t(
            "[P] Initializing enhanced dual-model LLM planner with modular VLM, combined model capability, classifier model support, and independent VLM probe support...")

        # MODEL CONFIGURATION (EASILY MODIFIABLE)
        # Core models
        self.reasoning_model = "qwen3-4b-reasoning_5"
        self.minispec_model = "qwen3-1.7b-writing_13"

        # NEW: Enhanced models for VLM and replan assessment
        self.vlm_model = "qwen2.5vl"
        self.replan_model = "qwen3:1.7b"

        # NEW: Combined model configuration
        self.combined_model = "qwen3-8b-combined_2"
        self.use_combined_model = False  # Default to dual-model approach

        # NEW: Probe model configuration (unfintuned model)
        self.probe_model = "qwen3:1.7b"

        # NEW: Classifier model configuration (for YOLO + LLM classification)
        self.classifier_model = "qwen3:0.6b"

        # Temperature settings for each model
        self.reasoning_temperature = 0.4  # Slightly higher for reasoning creativity
        self.minispec_temperature = 0.1  # Very low for precise code generation
        self.vlm_temperature = 0.3  # NEW: VLM temperature
        self.replan_temperature = 0.3  # NEW: Replan temperature
        self.combined_temperature = 0.1  # NEW: Combined model temperature
        self.probe_temperature = 0.1  # NEW: Probe model temperature
        self.classifier_temperature = 0.25  # NEW: Classifier model temperature

        # NEW: VLM configuration - MODULAR FLAGS
        self.vlm_enabled = True
        self.use_vlm_for_reasoning = False  # VLM for reasoning stage (dual-model approach)
        self.use_vlm_for_probe = True  # VLM for probe functionality (independent)

        print_t(f"[P] Reasoning model: {self.reasoning_model} (temp: {self.reasoning_temperature})")
        print_t(f"[P] MiniSpec model: {self.minispec_model} (temp: {self.minispec_temperature})")
        print_t(f"[P] VLM model: {self.vlm_model} (temp: {self.vlm_temperature}, enabled: {self.vlm_enabled})")
        print_t(f"[P] VLM for reasoning: {self.use_vlm_for_reasoning}")
        print_t(f"[P] VLM for probe: {self.use_vlm_for_probe}")
        print_t(f"[P] Replan model: {self.replan_model} (temp: {self.replan_temperature})")
        print_t(
            f"[P] Combined model: {self.combined_model} (temp: {self.combined_temperature}, enabled: {self.use_combined_model})")
        print_t(f"[P] Probe model: {self.probe_model} (temp: {self.probe_temperature})")
        print_t(f"[P] Classifier model: {self.classifier_model} (temp: {self.classifier_temperature})")

        # Initialize LLM instances
        self.reasoning_llm = LLMWrapper(temperature=self.reasoning_temperature)
        self.minispec_llm = LLMWrapper(temperature=self.minispec_temperature)
        self.vlm_llm = LLMWrapper(temperature=self.vlm_temperature)  # NEW: VLM instance
        self.replan_llm = LLMWrapper(temperature=self.replan_temperature)  # NEW: Replan instance
        self.combined_llm = LLMWrapper(temperature=self.combined_temperature)  # NEW: Combined instance
        self.probe_llm = LLMWrapper(temperature=self.probe_temperature)  # NEW: Probe instance
        self.classifier_llm = LLMWrapper(temperature=self.classifier_temperature)  # NEW: Classifier instance

        self.robot_type = robot_type
        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'

        self.type_folder_name = type_folder_name

        # Add CSV logger - automatically detects Windows Downloads folder
        self.csv_logger = SimpleCsvLogger()

        # Initialize inline prompts
        self._init_prompts()

        print_t(
            "[P] Enhanced dual-model planner with modular VLM, combined model capability, classifier model support, and independent VLM probe support initialized successfully")

    def _init_prompts(self):
        """Initialize all prompts as inline strings matching fine-tuning format"""
        print_t("[P] Loading enhanced inline prompts...")

        # STAGE 1: REASONING PROMPT - MATCHING FINE-TUNING FORMAT
        # This prompt will be used for both reasoning model and VLM reasoning
        self.reasoning_prompt = """You are a robot pilot reasoning system. Analyze the given scene and task to provide a concise reasoning sentence about what actions should be taken.
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
5. Adding re_plan will allow for persistent tasks such as following or if/then conditional tasks to proceed smoother
6. Use scan abstract when scanning for abstract objects such as food or toys
## Current Input
**Scene:** {scene_description}
**Task:** {task_description}
**Execution History:** {execution_history}
Provide your reasoning:"""

        # STAGE 2: MINISPEC WRITER PROMPT - MATCHING FINE-TUNING FORMAT
        self.minispec_prompt = """
You are a MiniSpec code generator. Given reasoning about a robot task, generate the appropriate MiniSpec program to execute the plan.
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

        # NEW: COMBINED MODEL PROMPT
        self.combined_prompt = """/no_think
You are a robot pilot. Generate a MiniSpec plan to fulfill the task or ask for clarification if the input is unclear.
## System Skills
High-level skills:
s (scan): Rotate to find object when not in current scene
sa (scan_abstract): Rotate to find abstract object by description when not in current scene
Low-level skills:
Movement: mf (move_forward), mb (move_backward), ml (move_left), mr (move_right), mu (move_up), md (move_down)
Rotation: tc (turn_cw), tu (turn_ccw)
Utilities: d (delay), tp (take_picture), rp (re_plan), t (time)
Vision: iv (is_visible), ox (object_x), oy (object_y), ow (object_width), oh (object_height), od (object_dis)
Logic: p (probe), l (log), g (goto)
Use high-level skills when available. Invoke skills using their short names. If a skill has no arguments, call it without parentheses.
## Input Format
- **Scene description:** Objects in current view with IDs, locations, and sizes (coordinates 0-1)
- **Task description:** Natural language instructions starting with "[A]" (action) or "[Q]" (question)
- **Execution history:** Previous actions taken (provided during replanning)
## Key Rules
1. Follow task description exactly - read word by word
2. Preserve directions mentioned in task (forward stays forward, left stays left)
3. Only use scene objects if task specifically asks for them
4. For "[A]" tasks: generate execution plan
5. For "[Q]" tasks: use 'log' to output the answer
6. During replanning: consider execution history and current scene
## Current Input
**Scene:** {scene_description}  
**Task:** {task_description}  
**Execution History:** {execution_history}
Generate response as a single MiniSpec program sentence."""

        # PROBE PROMPT - RESTORED
        self.probe_prompt = """/no_think
You are given a scene description and a question. You should output the answer to the question based on the scene description.
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

        # UPDATED: REPLAN ASSESSMENT PROMPT WITH REPLAN PROMPT FIELD
        self.replan_assessment_prompt = """Assess the current state of a robot task execution and determine next steps. Your goal is to aid the system in completing the orignal task thorugh judging task completion and providing valuable guidance.

Original Task: {original_task}
Past Reasoning: {past_reasoning}
Executed Commands: {executed_commands}
Current Scene: {current_scene}

Analyze the situation and provide:
1. Whether the task is complete, should continue, or needs a different approach
2. Specific feedback or guidance for next steps

General Guidelines:
1. If previous task involves scanning and no object found, suggest using scan abstract
2. If original command was to follow an object, decide if complete or an additional move prompt is necessary
3. If the original command has conditionals that were fulfilled replan for just the unfinished actions 

Provide assessment in this format:
Decision: COMPLETE/REPLAN/CONTINUE
Replan Prompt: [If replan, write a new task to help achieve the original task]
"""

        # Load additional data from files for reasoning stage
        try:
            with open(os.path.join(CURRENT_DIR, f"./assets/{self.type_folder_name}/guides.txt"), "r") as f:
                self.guides = f.read()
                print_t("[P] Loaded guides from file")
        except FileNotFoundError as e:
            print_t(f"[P] Warning: Could not load guides: {e}")
            self.guides = "Follow task instructions carefully and use appropriate skills."

    # ========== MODEL CONFIGURATION METHODS ==========
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

    def set_vlm_model(self, model_name: str, temperature: float = None, enabled: bool = True):
        """Set the VLM model and configuration"""
        self.vlm_model = model_name
        if temperature is not None:
            self.vlm_temperature = temperature
            self.vlm_llm = LLMWrapper(temperature=temperature)
        self.vlm_enabled = enabled
        print_t(f"[P] VLM model updated: {model_name} (temp: {self.vlm_temperature}, enabled: {enabled})")

    def set_replan_model(self, model_name: str, temperature: float = None):
        """Set the replan assessment model"""
        self.replan_model = model_name
        if temperature is not None:
            self.replan_temperature = temperature
            self.replan_llm = LLMWrapper(temperature=temperature)
        print_t(f"[P] Replan model updated: {model_name} (temp: {self.replan_temperature})")

    def set_probe_model(self, model_name: str, temperature: float = None):
        """Set the probe model (unfintuned model for scene questioning)"""
        self.probe_model = model_name
        if temperature is not None:
            self.probe_temperature = temperature
            self.probe_llm = LLMWrapper(temperature=temperature)
        print_t(f"[P] Probe model updated: {model_name} (temp: {self.probe_temperature})")

    def set_classifier_model(self, model_name: str, temperature: float = None):
        """Set the classifier model for YOLO + LLM classification"""
        self.classifier_model = model_name
        if temperature is not None:
            self.classifier_temperature = temperature
            self.classifier_llm = LLMWrapper(temperature=temperature)
        print_t(f"[P] Classifier model updated: {model_name} (temp: {self.classifier_temperature})")

    # ========== NEW: COMBINED MODEL CONFIGURATION METHODS ==========
    def set_combined_model(self, model_name: str, temperature: float = None):
        """Set the combined model and temperature"""
        self.combined_model = model_name
        if temperature is not None:
            self.combined_temperature = temperature
            self.combined_llm = LLMWrapper(temperature=temperature)
        print_t(f"[P] Combined model updated: {model_name} (temp: {self.combined_temperature})")

    def enable_combined_mode(self, enabled: bool = True):
        """Enable or disable combined model mode"""
        self.use_combined_model = enabled
        print_t(f"[P] Combined model mode: {'ENABLED' if enabled else 'DISABLED'}")
        print_t(f"[P] Planning approach: {'Single combined model' if enabled else 'Dual-model (reasoning + writing)'}")

    def disable_combined_mode(self):
        """Disable combined model mode (use dual-model approach)"""
        self.enable_combined_mode(False)

    def is_combined_mode_enabled(self) -> bool:
        """Check if combined model mode is currently enabled"""
        return self.use_combined_model

    def toggle_combined_mode(self) -> bool:
        """Toggle combined model mode on/off and return new state"""
        new_state = not self.use_combined_model
        self.enable_combined_mode(new_state)
        return new_state

    # ========== MODULAR VLM CONFIGURATION METHODS ==========
    def enable_vlm_reasoning(self, enabled: bool = True):
        """
        Enable/disable VLM for reasoning stage (dual-model approach only)

        Args:
            enabled: True = use VLM when image available (with LLM fallback)
                    False = always use LLM only for reasoning
        """
        self.use_vlm_for_reasoning = enabled
        self.vlm_enabled = enabled or self.use_vlm_for_probe  # Keep VLM enabled if probe needs it
        print_t(f"[P] VLM for reasoning: {'ENABLED' if enabled else 'DISABLED'}")

    def enable_vlm_probe(self, enabled: bool = True):
        """
        Enable/disable VLM for probe functionality (independent of reasoning)

        Args:
            enabled: True = use VLM for probe when image available (with fallback)
                    False = always use unfintuned model for probe
        """
        self.use_vlm_for_probe = enabled
        self.vlm_enabled = enabled or self.use_vlm_for_reasoning  # Keep VLM enabled if reasoning needs it
        print_t(f"[P] VLM for probe: {'ENABLED' if enabled else 'DISABLED'}")

    def is_vlm_reasoning_enabled(self) -> bool:
        """Check if VLM reasoning is currently enabled"""
        return self.use_vlm_for_reasoning and self.vlm_enabled

    def is_vlm_probe_enabled(self) -> bool:
        """Check if VLM probe is currently enabled"""
        return self.use_vlm_for_probe and self.vlm_enabled

    def toggle_vlm_reasoning(self) -> bool:
        """Toggle VLM reasoning on/off and return new state"""
        current = self.is_vlm_reasoning_enabled()
        self.enable_vlm_reasoning(not current)
        return not current

    def toggle_vlm_probe(self) -> bool:
        """Toggle VLM probe on/off and return new state"""
        current = self.is_vlm_probe_enabled()
        self.enable_vlm_probe(not current)
        return not current

    def disable_vlm_all(self):
        """Disable VLM for both reasoning and probe"""
        self.use_vlm_for_reasoning = False
        self.use_vlm_for_probe = False
        self.vlm_enabled = False
        print_t("[P] VLM disabled for both reasoning and probe")

    def enable_vlm_all(self):
        """Enable VLM for both reasoning and probe"""
        self.use_vlm_for_reasoning = True
        self.use_vlm_for_probe = True
        self.vlm_enabled = True
        print_t("[P] VLM enabled for both reasoning and probe")

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        """Initialize skillsets and vision"""
        print_t("[P] Initializing skillsets and vision...")
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill
        print_t("[P] Skillsets and vision initialized")

    # ========== NEW: COMBINED MODEL PLANNING METHOD ==========
    def plan_combined(self, task_description: str, scene_description: str,
                      execution_history: Optional[str] = None) -> str:
        """
        Single-model planning approach - generates MiniSpec directly from task/scene/history

        Args:
            task_description: Natural language task description
            scene_description: Current scene objects
            execution_history: Previous execution history (optional)

        Returns:
            str: MiniSpec code
        """
        print_t("[P] ====== COMBINED MODEL PLANNING ======")
        print_t(f"[P] Task: {task_description}")
        print_t(f"[P] Scene: {scene_description}")
        print_t(f"[P] History: {execution_history}")

        start_time = time.time()

        # Format the combined prompt
        prompt = self.combined_prompt.format(
            scene_description=scene_description,
            task_description=task_description,
            execution_history=execution_history or "None"
        )

        print_t(f"[P] Sending combined request to model: {self.combined_model}")
        print_t(f"[P] Combined prompt length: {len(prompt)} characters")
        print_t(f"[P] Combined prompt preview: {prompt[:200]}...")

        try:
            # Make the LLM call
            response = self.combined_llm.request(prompt, self.combined_model, stream=False)

            combined_time = time.time() - start_time
            print_t(f"[P] Combined planning completed in {combined_time:.2f}s")
            print_t(f"[P] Raw combined response: {response}")

            # Extract code (remove any prefixes if present)
            if response.startswith("Response:"):
                code = response[9:].strip()
            else:
                code = response.strip()

            # Remove trailing semicolon if present
            if code.endswith(';'):
                code = code[:-1]

            print_t(f"[P] Extracted combined MiniSpec code: {code}")
            return code

        except Exception as e:
            print_t(f"[P] Error in combined planning: {e}")
            raise e

    def plan_reasoning(self, task_description: str, scene_description: str,
                       execution_history: Optional[str] = None, image_path: Optional[str] = None,
                       force_vlm: bool = None) -> Tuple[str, str]:
        """
        Stage 1: Generate reasoning using either reasoning model or VLM

        Args:
            task_description: Natural language task description
            scene_description: Current scene objects
            execution_history: Previous execution history (optional)
            image_path: Camera image for VLM reasoning (optional)
            force_vlm: Override to force VLM usage (optional)

        Returns:
            Tuple[str, str]: (reasoning_text, model_used)
        """
        print_t("[P] ====== STAGE 1: REASONING ======")
        print_t(f"[P] Task: {task_description}")
        print_t(f"[P] Scene: {scene_description}")
        print_t(f"[P] History: {execution_history}")
        print_t(f"[P] Image available: {image_path is not None}")

        # Determine which model to use
        should_use_vlm = force_vlm if force_vlm is not None else (
                self.use_vlm_for_reasoning and self.vlm_enabled and image_path is not None
        )

        if should_use_vlm:
            print_t(f"[P] Using VLM for reasoning: {self.vlm_model}")
            model_to_use = self.vlm_model
            llm_to_use = self.vlm_llm
            model_type = "VLM"
        else:
            print_t(f"[P] Using reasoning model: {self.reasoning_model}")
            model_to_use = self.reasoning_model
            llm_to_use = self.reasoning_llm
            model_type = "REASONING"

        start_time = time.time()

        # Use same prompt format for both models
        prompt = self.reasoning_prompt.format(
            scene_description=scene_description,
            task_description=task_description,
            execution_history=execution_history or "None"
        )

        print_t(f"[P] Sending reasoning request to model: {model_to_use}")
        print_t(f"[P] Reasoning prompt length: {len(prompt)} characters")
        print_t(f"[P] Reasoning prompt preview: {prompt[:200]}...")

        try:
            # Make the LLM call (with image if using VLM)
            if should_use_vlm and image_path:
                response = llm_to_use.request(
                    prompt=prompt,
                    model_name=model_to_use,
                    image_path=image_path,
                    stream=False
                )
            else:
                response = llm_to_use.request(prompt, model_to_use, stream=False)

            reasoning_time = time.time() - start_time
            print_t(f"[P] Reasoning completed in {reasoning_time:.2f}s using {model_type}")
            print_t(f"[P] Raw reasoning response: {response}")

            # Extract reasoning (remove "Reason:" prefix if present)
            if response.startswith("Reason:"):
                reasoning = response[7:].strip()
            else:
                reasoning = response.strip()

            print_t(f"[P] Extracted reasoning: {reasoning}")
            return reasoning, model_to_use

        except Exception as e:
            print_t(f"[P] Error in {model_type} reasoning: {e}")
            # Fallback to the other model if available
            if should_use_vlm and not force_vlm:
                print_t("[P] Falling back to reasoning model due to VLM error")
                return self.plan_reasoning(task_description, scene_description, execution_history, None, False)
            else:
                raise e

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
             execution_history: Optional[str] = None, image_path: Optional[str] = None,
             use_vlm_for_reasoning: bool = None) -> Tuple[str, str]:
        """
        Enhanced main planning method with combined model OR dual-model capability

        Args:
            task_description: Natural language task description
            scene_description: Current scene objects (optional)
            error_message: Error context (optional)
            execution_history: Previous execution history (optional)
            image_path: Camera image for VLM reasoning (optional)
            use_vlm_for_reasoning: Override VLM usage for reasoning (None=auto, True=force VLM, False=force reasoning model)

        Returns:
            Tuple[str, str]: (reasoning, minispec_code)
        """
        print_t("[P] ========== ENHANCED PLANNING START ==========")
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
            # NEW: Route to combined model OR dual-model approach
            if self.use_combined_model:
                print_t("[P] ========== USING COMBINED MODEL APPROACH ==========")

                # Single model does both reasoning and code generation
                code = self.plan_combined(task_description, scene_description, execution_history)

                # For logging purposes, set reasoning to a descriptive message
                reasoning = f"Combined model approach - direct code generation from task"
                actual_model_used = self.combined_model

                total_time = time.time() - total_start_time
                print_t(f"[P] ========== COMBINED MODEL PLANNING COMPLETE in {total_time:.2f}s ==========")
                print_t(f"[P] Generated MiniSpec code: {code}")
                print_t(f"[P] Model used: {actual_model_used}")

                # Log successful operation
                self.csv_logger.log(
                    user_input=task_description,
                    model1_reasoning=reasoning,
                    model2_minispec=code,
                    execution_time=total_time,
                    reasoning_model=actual_model_used,
                    minispec_model="N/A (Combined)",
                    reasoning_temperature=self.combined_temperature,
                    minispec_temperature=0,
                    errors=error_message or "",
                    combined_mode=True
                )

                return reasoning, code

            else:
                print_t("[P] ========== USING DUAL-MODEL APPROACH ==========")

                # Stage 1: Generate reasoning (using either reasoning model or VLM)
                reasoning, actual_reasoning_model = self.plan_reasoning(
                    task_description, scene_description, execution_history, image_path, use_vlm_for_reasoning
                )

                # Stage 2: Generate MiniSpec code (gets ONLY reasoning)
                code = self.plan_minispec(reasoning)

                total_time = time.time() - total_start_time
                print_t(f"[P] ========== DUAL-MODEL PLANNING COMPLETE in {total_time:.2f}s ==========")
                print_t(f"[P] Final reasoning: {reasoning}")
                print_t(f"[P] Final MiniSpec code: {code}")
                print_t(f"[P] Actual reasoning model used: {actual_reasoning_model}")

                # Log successful operation with actual model used
                actual_reasoning_temperature = (
                    self.vlm_temperature if actual_reasoning_model == self.vlm_model
                    else self.reasoning_temperature
                )

                self.csv_logger.log(
                    user_input=task_description,
                    model1_reasoning=reasoning,
                    model2_minispec=code,
                    execution_time=total_time,
                    reasoning_model=actual_reasoning_model,
                    minispec_model=self.minispec_model,
                    reasoning_temperature=actual_reasoning_temperature,
                    minispec_temperature=self.minispec_temperature,
                    errors=error_message or "",
                    combined_mode=False
                )

                return reasoning, code

        except Exception as e:
            print_t(f"[P] ERROR in enhanced planning: {e}")
            print_t(f"[P] Returning fallback response")

            # Log failed operation
            self.csv_logger.log(
                user_input=task_description,
                model1_reasoning=f"ERROR: {e}",
                model2_minispec="log('Planning failed')",
                execution_time=time.time() - total_start_time,
                reasoning_model=self.combined_model if self.use_combined_model else self.reasoning_model,
                minispec_model="N/A (Combined)" if self.use_combined_model else self.minispec_model,
                reasoning_temperature=self.combined_temperature if self.use_combined_model else self.reasoning_temperature,
                minispec_temperature=0 if self.use_combined_model else self.minispec_temperature,
                errors=str(e),
                combined_mode=self.use_combined_model
            )

            return f"Error in planning: {e}", "log('Planning failed')"

    # ========== REPLAN ASSESSMENT WITH NATURAL LANGUAGE RESTART ==========
    def assess_replan(self, original_task: str, past_reasoning: str, executed_commands: str,
                      current_scene: str) -> Tuple[str, int, str, str]:
        """Enhanced replan assessment - generates new natural language task for restart"""
        print_t(f"[P] === REPLAN ASSESSMENT ===")
        print_t(f"[P] Original task: {original_task}")
        print_t(f"[P] Current scene: {current_scene}")

        try:
            prompt = self.replan_assessment_prompt.format(
                original_task=original_task,
                past_reasoning=past_reasoning,
                executed_commands=executed_commands,
                current_scene=current_scene
            )

            print_t(f"[P] Sending replan assessment to model: {self.replan_model}")
            response = self.replan_llm.request(prompt, self.replan_model, stream=False)

            print_t(f"[P] Replan assessment response: {response}")

            # Parse response
            lines = response.strip().split('\n')
            decision = "CONTINUE"
            progress = 50
            feedback = response
            replan_prompt = ""

            for line in lines:
                if 'Decision:' in line:
                    decision = line.split(':', 1)[1].strip()
                elif 'Progress:' in line:
                    try:
                        progress = int(line.split(':', 1)[1].strip().replace('%', ''))
                    except:
                        pass
                elif 'Feedback:' in line:
                    feedback = line.split(':', 1)[1].strip()
                elif 'Replan Prompt:' in line:
                    replan_prompt = line.split(':', 1)[1].strip()

            print_t(f"[P] Parsed assessment: {decision}, {progress}%, {feedback}")
            if replan_prompt:
                print_t(f"[P] New task for restart: {replan_prompt}")

            return decision, progress, feedback, replan_prompt

        except Exception as e:
            print_t(f"[P] Error in replan assessment: {e}")
            return "CONTINUE", 50, "Assessment failed, continuing", ""

    # PROBE FUNCTIONALITY WITH MODULAR VLM SUPPORT
    def probe(self, question: str) -> Tuple[MiniSpecValueType, bool]:
        """Probe method for scene questioning using VLM (if enabled for probe) or unfintuned fallback model"""
        print_t(f"[P] === PROBE REQUEST ===")
        print_t(f"[P] Question: {question}")

        # Get current image for VLM probe if VLM probe is enabled
        current_image_path = None
        should_use_vlm = self.vlm_enabled and self.use_vlm_for_probe  # Use separate probe flag

        if should_use_vlm:
            # Try to get current frame - this needs to be called from controller context
            try:
                # This will be set by the controller when it calls probe
                if hasattr(self, '_current_image_path') and self._current_image_path:
                    current_image_path = self._current_image_path
                    print_t(f"[P] Using current image for VLM probe: {current_image_path}")
                else:
                    print_t(f"[P] No current image available, falling back to unfintuned model")
                    should_use_vlm = False
            except Exception as e:
                print_t(f"[P] Error getting image for VLM probe: {e}, falling back to unfintuned model")
                should_use_vlm = False

        # Format probe prompt
        prompt = self.probe_prompt.format(
            scene_description=self.vision_skill.get_obj_list(),
            question=question
        )

        print_t(f"[P] Probe prompt length: {len(prompt)} characters")

        try:
            if should_use_vlm and current_image_path:
                print_t(f"[P] Sending probe to VLM model: {self.vlm_model}")
                response = self.vlm_llm.request(
                    prompt=prompt,
                    model_name=self.vlm_model,
                    image_path=current_image_path,
                    stream=False
                )
                print_t(f"[P] VLM probe response: {response}")
            else:
                print_t(f"[P] Sending probe to unfintuned probe model: {self.probe_model}")
                response = self.probe_llm.request(prompt, self.probe_model)
                print_t(f"[P] Unfintuned probe response: {response}")

            return evaluate_value(response), False

        except Exception as e:
            print_t(f"[P] Error in probe request: {e}")
            # Fallback to unfintuned model if VLM fails
            if should_use_vlm:
                print_t(f"[P] VLM probe failed, falling back to unfintuned model: {self.probe_model}")
                try:
                    response = self.probe_llm.request(prompt, self.probe_model)
                    print_t(f"[P] Fallback probe response: {response}")
                    return evaluate_value(response), False
                except Exception as fallback_e:
                    print_t(f"[P] Fallback probe also failed: {fallback_e}")
                    return evaluate_value("False"), False
            else:
                return evaluate_value("False"), False