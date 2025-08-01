from PIL import Image
import queue, time, os, json
from typing import Optional, Tuple
import asyncio
import uuid
from enum import Enum
import re
from threading import Event
from .shared_frame import SharedFrame, Frame
from .yolo_client import YoloClient
from .yolo_grpc_client import YoloGRPCClient
from .yoloe_client import YoloEClient
from .simple_out_of_frame_corrector import SimpleOutOfFrameCorrector
from .simplified_enhanced_replan_controller import SimplifiedEnhancedReplanController, PipelineState, \
    ReplanDecision  # UPDATED: Simplified version
from .tello_wrapper import TelloWrapper
from .virtual_robot_wrapper import VirtualRobotWrapper
from .abs.robot_wrapper import RobotWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import print_t, input_t
from .minispec_interpreter import MiniSpecInterpreter, Statement
from .abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class LLMController():
    def __init__(self, robot_type, use_http=False, message_queue: Optional[queue.Queue] = None):
        print_t(
            "[C] Initializing LLMController with dual-model, modular VLM, YoloE, YOLO+LLM classification, auto-correction, simplified enhanced replan controller, probe support, and combined model capability...")

        self.shared_frame = SharedFrame()
        if use_http:
            self.yolo_client = YoloClient(shared_frame=self.shared_frame)
        else:
            self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)

        # YoloE client for visual prompt detection
        self.yoloe_client = YoloEClient(shared_frame=self.shared_frame)

        self.vision = VisionSkillWrapper(self.shared_frame)

        # Simple auto-correction system (bowling bumpers)
        self.auto_corrector = SimpleOutOfFrameCorrector(self.vision)

        self.latest_frame = None
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue

        # ADD CANCELLATION SUPPORT
        self.cancellation_event = Event()
        self.current_interpreter = None

        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            self.cache_folder = message_queue.get()

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        match robot_type:
            case RobotType.TELLO:
                print_t("[C] Start Tello drone...")
                self.drone: RobotWrapper = TelloWrapper()
            case RobotType.GEAR:
                print_t("[C] Start Gear robot car...")
                from .gear_wrapper import GearWrapper
                self.drone: RobotWrapper = GearWrapper()
            case _:
                print_t("[C] Start virtual drone...")
                self.drone: RobotWrapper = VirtualRobotWrapper()

        self.planner = LLMPlanner(robot_type)

        # NEW: Simplified enhanced replan controller with only COMPLETE/CONTINUE decisions
        self.replan_controller = SimplifiedEnhancedReplanController(self.planner.reasoning_llm)

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("move_forward", self.drone.move_forward, "Move forward by a distance",
                              args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("move_backward", self.drone.move_backward, "Move backward by a distance",
                              args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("move_left", self.drone.move_left, "Move left by a distance",
                              args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("move_right", self.drone.move_right, "Move right by a distance",
                              args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance",
                              args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees",
                              args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees",
                              args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.skill_delay, "Wait for specified seconds",
                                                            args=[SkillArg("seconds", float)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object",
                              args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)",
                              args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)",
                              args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("object_width", self.vision.object_width, "Get object's width in (0,1)",
                              args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("object_height", self.vision.object_height, "Get object's height in (0,1)",
                              args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("object_dis", self.vision.object_distance, "Get object's distance in cm",
                              args=[SkillArg("object_name", str)]))

        # ========== PROBE FUNCTIONALITY WITH VLM SUPPORT ==========
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("probe", self.skill_probe, "Probe the LLM for reasoning with VLM support",
                              args=[SkillArg("question", str)]))

        # ========== YOLOE FUNCTIONALITY ==========
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("ye", self.skill_yoloe_detect, "YoloE visual prompt detection",
                              args=[SkillArg("description", str)]))

        # ========== NEW: YOLO + LLM CLASSIFICATION FUNCTIONALITY ==========
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("yolo_classify", self.skill_yolo_classify,
                              "YOLO + LLM object classification with position",
                              args=[SkillArg("description", str)]))

        self.low_level_skillset.add_skill(
            LowLevelSkillItem("log", self.skill_log, "Output text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("take_picture", self.skill_take_picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("re_plan", self.skill_re_plan, "Replanning"))

        self.low_level_skillset.add_skill(LowLevelSkillItem("goto", self.skill_goto, "goto the object",
                                                            args=[SkillArg("object_name[*x-value]", str)]))
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("time", self.skill_time, "Get current execution time", args=[]))

        # load high-level skills
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)

        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'
        with open(os.path.join(CURRENT_DIR, f"assets/{type_folder_name}/high_level_skills.json"), "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))

        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = self.high_level_skillset

        # CRITICAL: Connect cancellation event to robot wrapper
        self.drone.set_cancellation_event(self.cancellation_event)

        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset,
                          vision_skill=self.vision)

        self.current_plan = None
        self.execution_history = None
        self.execution_time = time.time()

        # Enhanced pipeline state tracking
        self.task_start_time = None
        self.initial_scene = None
        self.pipeline_iterations = []

        print_t(
            "[C] LLMController with dual-model, modular VLM, YoloE, YOLO+LLM classification, auto-correction, simplified enhanced replan controller, probe support, and combined model capability initialization complete")

    # ========== MODEL CONFIGURATION METHODS ==========
    def set_reasoning_model(self, model_name: str, temperature: float = None):
        """Configure the reasoning model"""
        print_t(f"[C] Setting reasoning model to: {model_name}")
        self.planner.set_reasoning_model(model_name, temperature)

    def set_minispec_model(self, model_name: str, temperature: float = None):
        """Configure the MiniSpec generation model"""
        print_t(f"[C] Setting MiniSpec model to: {model_name}")
        self.planner.set_minispec_model(model_name, temperature)

    def set_vlm_model(self, model_name: str, temperature: float = None, enabled: bool = True):
        """Configure the VLM model"""
        print_t(f"[C] Setting VLM model to: {model_name} (enabled: {enabled})")
        self.planner.set_vlm_model(model_name, temperature, enabled)

    def set_replan_model(self, model_name: str, temperature: float = None):
        """Configure the replan assessment model"""
        print_t(f"[C] Setting replan model to: {model_name}")
        self.planner.set_replan_model(model_name, temperature)
        self.replan_controller.set_replan_model(model_name, temperature)

    def set_probe_model(self, model_name: str, temperature: float = None):
        """Configure the probe model (unfintuned model for scene questioning)"""
        print_t(f"[C] Setting probe model to: {model_name}")
        self.planner.set_probe_model(model_name, temperature)

    def set_classifier_model(self, model_name: str, temperature: float = None):
        """Configure the YOLO+LLM classifier model"""
        print_t(f"[C] Setting classifier model to: {model_name}")
        self.planner.set_classifier_model(model_name, temperature)

    # ========== NEW: COMBINED MODEL CONFIGURATION METHODS ==========
    def set_combined_model(self, model_name: str, temperature: float = None):
        """Configure the combined model"""
        print_t(f"[C] Setting combined model to: {model_name}")
        self.planner.set_combined_model(model_name, temperature)

    def enable_combined_mode(self, enabled: bool = True):
        """Enable or disable combined model mode"""
        print_t(f"[C] Combined model mode: {'ENABLING' if enabled else 'DISABLING'}")
        self.planner.enable_combined_mode(enabled)

    def disable_combined_mode(self):
        """Disable combined model mode (use dual-model approach)"""
        print_t(f"[C] Combined model mode: DISABLING")
        self.planner.disable_combined_mode()

    def is_combined_mode_enabled(self) -> bool:
        """Check if combined model mode is currently enabled"""
        return self.planner.is_combined_mode_enabled()

    def toggle_combined_mode(self) -> bool:
        """Toggle combined model mode on/off and return new state"""
        new_state = self.planner.toggle_combined_mode()
        print_t(f"[C] Combined model mode toggled: {'ON' if new_state else 'OFF'}")
        return new_state

    def enable_vlm(self, enabled: bool = True):
        """Enable or disable VLM model (sets vlm_enabled only - for backwards compatibility)"""
        print_t(f"[C] VLM model {'enabled' if enabled else 'disabled'}")
        self.planner.set_vlm_model(self.planner.vlm_model, enabled=enabled)

    # ========== MODULAR VLM REASONING AND PROBE CONFIGURATION ==========
    def enable_vlm_reasoning(self, enabled: bool = True):
        """
        Enable/disable VLM for reasoning stage (dual-model approach only)

        Args:
            enabled: True = use VLM when image available (with LLM fallback)
                    False = always use LLM only for reasoning
        """
        print_t(f"[C] VLM reasoning: {'ENABLING' if enabled else 'DISABLING'}")
        self.planner.enable_vlm_reasoning(enabled)

    def enable_vlm_probe(self, enabled: bool = True):
        """
        Enable/disable VLM for probe functionality (independent of reasoning)

        Args:
            enabled: True = use VLM for probe when image available (with fallback)
                    False = always use unfintuned model for probe
        """
        print_t(f"[C] VLM probe: {'ENABLING' if enabled else 'DISABLING'}")
        self.planner.enable_vlm_probe(enabled)

    def is_vlm_reasoning_enabled(self) -> bool:
        """Check if VLM reasoning is currently enabled"""
        return self.planner.is_vlm_reasoning_enabled()

    def is_vlm_probe_enabled(self) -> bool:
        """Check if VLM probe is currently enabled"""
        return self.planner.is_vlm_probe_enabled()

    def toggle_vlm_reasoning(self) -> bool:
        """Toggle VLM reasoning on/off and return new state"""
        new_state = self.planner.toggle_vlm_reasoning()
        print_t(f"[C] VLM reasoning toggled: {'ON' if new_state else 'OFF'}")
        return new_state

    def toggle_vlm_probe(self) -> bool:
        """Toggle VLM probe on/off and return new state"""
        new_state = self.planner.toggle_vlm_probe()
        print_t(f"[C] VLM probe toggled: {'ON' if new_state else 'OFF'}")
        return new_state

    def disable_vlm_all(self):
        """Disable VLM for both reasoning and probe"""
        print_t(f"[C] Disabling VLM for both reasoning and probe")
        self.planner.disable_vlm_all()

    def enable_vlm_all(self):
        """Enable VLM for both reasoning and probe"""
        print_t(f"[C] Enabling VLM for both reasoning and probe")
        self.planner.enable_vlm_all()

    # ========== SIMPLIFIED ENHANCED REPLAN CONTROLLER CONFIGURATION ==========
    def configure_safety_limits(self, max_iterations: int = None, max_time: float = None,
                                max_failures: int = None):
        """Configure simplified enhanced replan controller safety limits"""
        self.replan_controller.configure_safety_limits(max_iterations, max_time, max_failures)

    # ========== YOLOE CONFIGURATION METHODS ==========
    def set_yoloe_confidence(self, threshold: float):
        """Configure YoloE confidence threshold"""
        print_t(f"[C] Setting YoloE confidence threshold to: {threshold}")
        self.yoloe_client.set_confidence_threshold(threshold)

    def clear_yoloe_cache(self):
        """Clear YoloE reference image cache"""
        print_t("[C] Clearing YoloE reference cache")
        self.yoloe_client.clear_reference_cache()

    # ========== AUTO-CORRECTION CONFIGURATION METHODS ==========
    def enable_auto_correction(self, enabled: bool = True):
        """Enable or disable auto-correction (bowling bumpers)"""
        if enabled:
            self.auto_corrector.enable()
        else:
            self.auto_corrector.disable()
        print_t(f"[C] Auto-correction {'enabled' if enabled else 'disabled'}")

    def set_correction_thresholds(self, edge_threshold: float = None, size_small: float = None,
                                  size_large: float = None):
        """Configure auto-correction sensitivity thresholds"""
        self.auto_corrector.set_thresholds(edge_threshold, size_small, size_large)

    def set_correction_amounts(self, turn_degrees: int = None, move_distance: int = None):
        """Configure auto-correction movement amounts"""
        self.auto_corrector.set_correction_amounts(turn_degrees, move_distance)

    # ========== SKILL METHODS WITH CANCELLATION SUPPORT ==========
    def skill_time(self) -> Tuple[float, bool]:
        return time.time() - self.execution_time, False

    def skill_yoloe_detect(self, description: str) -> Tuple[Optional[str], bool]:
        """
        YoloE visual prompt detection skill

        Args:
            description: Text description of object to find

        Returns:
            Tuple of (detection_result, replan_needed)
            - detection_result: "object_name[x_value]" if found, "False" if not found
            - replan_needed: True if detection failed and should replan
        """
        print_t(f"[C] === YOLOE DETECTION SKILL ===")
        print_t(f"[C] YoloE searching for: {description}")

        # Check for cancellation before executing
        if self.cancellation_event.is_set():
            print_t("[C] YoloE detection cancelled")
            return "False", False

        # Set target for auto-correction if object is found
        self.auto_corrector.set_target_object(description)

        try:
            # Get current frame
            current_frame = Frame(self.latest_frame)

            # Perform YoloE detection
            result, replan_needed = self.yoloe_client.detect_by_description(
                current_frame,
                description,
                conf=self.yoloe_client.confidence_threshold
            )

            if result is not None:
                print_t(f"[C] YoloE detection successful: {result}")
                # Set the specific found object for auto-correction
                self.auto_corrector.set_target_object(result)
                return result, replan_needed
            else:
                print_t(f"[C] YoloE detection failed for: {description}")
                self.auto_corrector.clear_target()
                return "False", replan_needed

        except Exception as e:
            print_t(f"[C] Error in YoloE detection: {e}")
            self.auto_corrector.clear_target()
            return "False", True  # Replan due to error

    def skill_yolo_classify(self, description: str) -> Tuple[Optional[str], bool]:
        """
        YOLO + LLM classification skill for scan_abstract

        Returns positioned object reference for goto compatibility
        Example: "food" → "apple[0.7]" → can be used by goto(_1)
        """
        print_t(f"[C] === YOLO + LLM CLASSIFICATION ===")
        print_t(f"[C] Classifying for: {description}")

        # Check for cancellation before executing
        if self.cancellation_event.is_set():
            print_t("[C] YOLO classification cancelled")
            return "False", False

        try:
            # Get current scene object list from YOLO
            scene_objects = self.vision.get_obj_list()

            if not scene_objects or scene_objects == "[]":
                print_t(f"[C] No objects detected for classification")
                return "False", False

            print_t(f"[C] Available objects: {scene_objects}")

            # Classification prompt - ask LLM to identify which object matches description
            classification_prompt = f"""/no think You are an object classifier. You will be given an object list of object names and their coordinates, determine if it any match the given criteria. 
            Objects will appear in 2d, ignore this and determine if what they represent matches the criteria.

Detected Objects: {scene_objects}
Criteria: {description}

Instructions:
- Return ONLY the exact object name that best matches
- If multiple objects match, pick the most relevant one
- If no objects match, return "False"
- For status questions such as hot or cold, guess off of object attributes and be categorical in your assumptions. (For example a stove would be hot, a refrigerator would be cold, any animals would be alive, any bowls/chairs/etc. would be stationary)
- Format you answer in this format: <answer> (False/Object_name) </answer>
"""

            print_t(f"[C] Sending to classifier model: {self.planner.classifier_model}")

            # Get classification from LLM
            response = self.planner.classifier_llm.request(
                classification_prompt,
                self.planner.classifier_model,
                stream=False
            )

            # Clean up response - robust parsing for multiple formats
            response_clean = response.strip()

            # Extract from answer tags if present
            if '<answer>' in response_clean and '</answer>' in response_clean:
                # Extract content between <answer> and </answer>
                start = response_clean.find('<answer>') + len('<answer>')
                end = response_clean.find('</answer>')
                response_clean = response_clean[start:end].strip()
                print_t(f"[C] Extracted from answer tags: '{response_clean}'")

            # Remove thinking tags if they still exist (fallback)
            elif '<think>' in response_clean:
                if '</think>' in response_clean:
                    response_clean = response_clean.split('</think>')[-1].strip()
                else:
                    response_clean = response_clean.split('<think>')[-1].strip()
                print_t(f"[C] Extracted after think tags: '{response_clean}'")

            # Clean up parentheses and quotes
            response_clean = response_clean.strip('()').strip('"\'').strip()

            # Robust parsing for different formats
            if response_clean.lower() == 'false':
                # Format: (False)
                classified_object = 'false'
                print_t(f"[C] No match found (False)")
            elif response_clean.lower().startswith('true/'):
                # Format: (True/person_1)
                classified_object = response_clean[5:].strip().lower()  # Remove "True/" prefix
                print_t(f"[C] Found match with True/ prefix: '{classified_object}'")
            elif response_clean.lower() != 'false' and len(response_clean) > 0:
                # Format: (person_1) - direct object name
                classified_object = response_clean.lower()
                print_t(f"[C] Found direct object name: '{classified_object}'")
            else:
                # Fallback for empty or malformed responses
                classified_object = 'false'
                print_t(f"[C] Fallback to False for malformed response")

            print_t(f"[C] Raw response: '{response}'")
            print_t(f"[C] Final classified object: '{classified_object}'")

            if classified_object == "false" or not classified_object:
                print_t(f"[C] No object matches '{description}'")
                return "False", False

            # Get the position of the classified object using existing vision system
            print_t(f"[C] Looking up position for: {classified_object}")
            object_x_result, position_error = self.vision.object_x(classified_object)

            if position_error or isinstance(object_x_result, str):
                print_t(f"[C] Could not find position for '{classified_object}': {object_x_result}")
                return "False", False

            # Format as positioned object reference for goto compatibility
            positioned_object = f"{classified_object}[{object_x_result:.2f}]"
            print_t(f"[C] YOLO+LLM classification successful: {positioned_object}")

            # Set target for auto-correction system
            self.auto_corrector.set_target_object(classified_object)

            return positioned_object, False

        except Exception as e:
            print_t(f"[C] Error in YOLO+LLM classification: {e}")
            self.auto_corrector.clear_target()
            return "False", True  # Signal replan needed

    def skill_goto(self, object_name: str) -> Tuple[None, bool]:
        print_t(f'[C] Executing goto: {object_name}')

        # Check for cancellation before executing
        if self.cancellation_event.is_set():
            print_t("[C] GOTO cancelled before execution")
            return None, False

        # Set target for auto-correction
        base_name = object_name.split('[')[0] if '[' in object_name else object_name
        self.auto_corrector.set_target_object(base_name)

        if '[' in object_name:
            x = float(object_name.split('[')[1].split(']')[0])
        else:
            x = self.vision.object_x(object_name)[0]

        print_t(f'[C] GOTO target x-coordinate: {x} (type: {type(x)})')

        if x > 0.55:
            turn_amount = int((x - 0.5) * 70)
            print_t(f'[C] Turning clockwise {turn_amount} degrees')
            if not self.cancellation_event.is_set():
                self.drone.turn_cw(turn_amount)
        elif x < 0.45:
            turn_amount = int((0.5 - x) * 70)
            print_t(f'[C] Turning counter-clockwise {turn_amount} degrees')
            if not self.cancellation_event.is_set():
                self.drone.turn_ccw(turn_amount)

        print_t('[C] Moving forward 110 units')
        if not self.cancellation_event.is_set():
            self.drone.move_forward(110)
        return None, False

    def skill_take_picture(self) -> Tuple[None, bool]:
        # Check for cancellation before executing
        if self.cancellation_event.is_set():
            print_t("[C] Take picture cancelled")
            return None, False

        img_path = os.path.join(self.cache_folder, f"{uuid.uuid4()}.jpg")
        Image.fromarray(self.latest_frame).save(img_path)
        print_t(f"[C] Picture saved to: {img_path}")
        self.append_message((img_path,))
        return None, False

    def skill_log(self, text: str) -> Tuple[None, bool]:
        log_message = f"[LOG] {text}"
        self.append_message(log_message)
        print_t(log_message)
        return None, False

    def skill_re_plan(self) -> Tuple[None, bool]:
        print_t("[C] Re-planning requested")
        return None, True

    def skill_delay(self, s: float) -> Tuple[None, bool]:
        print_t(f"[C] Delaying for {s} seconds")

        # Check for cancellation during delay
        start_time = time.time()
        while time.time() - start_time < s:
            if self.cancellation_event.is_set():
                print_t("[C] Delay cancelled")
                return None, False
            time.sleep(0.1)
        return None, False

    def skill_probe(self, question: str) -> Tuple[any, bool]:
        """Probe skill with modular VLM support - provides current image to planner if VLM probe enabled"""
        print_t(f"[C] === PROBE SKILL WITH MODULAR VLM SUPPORT ===")
        print_t(f"[C] Probe question: {question}")

        # Check for cancellation before executing
        if self.cancellation_event.is_set():
            print_t("[C] Probe cancelled")
            return "False", False

        try:
            # Get current frame path for VLM probe if VLM probe is enabled (separate from reasoning)
            if self.planner.is_vlm_probe_enabled():
                current_image_path = self.get_latest_frame_path()
                if current_image_path:
                    print_t(f"[C] Providing current image for VLM probe: {current_image_path}")
                    # Set current image path for planner to use
                    self.planner._current_image_path = current_image_path
                else:
                    print_t(f"[C] Could not get current image, probe will use unfintuned fallback")
                    self.planner._current_image_path = None
            else:
                print_t(f"[C] VLM probe disabled, using unfintuned model")
                self.planner._current_image_path = None

            # Call planner's probe method
            result, replan_needed = self.planner.probe(question)

            # Clean up image path
            if hasattr(self.planner, '_current_image_path'):
                delattr(self.planner, '_current_image_path')

            print_t(f"[C] Probe result: {result}")
            return result, replan_needed

        except Exception as e:
            print_t(f"[C] Error in probe skill: {e}")
            # Clean up image path on error
            if hasattr(self.planner, '_current_image_path'):
                delattr(self.planner, '_current_image_path')
            return "False", False

    # ========== AUTO-CORRECTION INTEGRATION ==========
    def _extract_target_objects_from_plan(self, minispec_code: str) -> Optional[str]:
        """
        Extract target object names from MiniSpec code for auto-correction
        Simple pattern matching for common skills that work with objects

        Args:
            minispec_code: The MiniSpec program code

        Returns:
            Object name if found, None otherwise
        """
        # Look for common patterns that indicate target objects
        patterns = [
            r"sa\('([^']+)'\)",  # scan_abstract('object')
            r"s\('([^']+)'\)",  # scan('object')
            r"g\('([^']+)'\)",  # goto('object')
            r"ye\('([^']+)'\)",  # yoloe_detect('object')
            r"yc\('([^']+)'\)",  # yolo_classify('object')
        ]

        for pattern in patterns:
            match = re.search(pattern, minispec_code)
            if match:
                target_object = match.group(1)
                print_t(f"[C] Detected target object for auto-correction: {target_object}")
                return target_object

        return None

    def _apply_auto_correction_if_needed(self):
        """
        Apply auto-correction during planning downtime
        Called AFTER complete skill execution, BEFORE next planning iteration
        """
        try:
            print_t("[C] --- Auto-Correction Check (Planning Downtime) ---")

            # Small delay to let vision system update
            time.sleep(0.3)

            # Apply auto-correction (like bowling bumpers)
            correction_applied = self.auto_corrector.check_and_correct(self.drone)

            if correction_applied:
                print_t("[C] Auto-correction applied successfully")
                self.append_message("[BUMPER] Auto-correction applied")
                # Brief pause after correction
                time.sleep(0.5)
                return True
            else:
                print_t("[C] No auto-correction needed")
                return False

        except Exception as e:
            print_t(f"[C] Error in auto-correction: {e}")
            # Auto-correction errors should not break the main execution
            return False

    # ========== SIMPLIFIED PIPELINE STATE MANAGEMENT ==========
    def _create_pipeline_state(self, planning_iteration: int, original_task: str,
                               plan_reason: str, minispec_code: str, planning_time: float,
                               execution_result: any, execution_time: float, execution_success: bool,
                               corrections_applied: bool) -> PipelineState:
        """
        Create comprehensive pipeline state for simplified replan assessment
        """
        current_time = time.time()
        total_execution_time = current_time - self.task_start_time if self.task_start_time else 0

        # Get current scene information
        current_scene = self.vision.get_obj_list()

        # Calculate scene changes
        scene_changes = "No significant changes detected"
        if self.initial_scene and current_scene != self.initial_scene:
            scene_changes = f"Scene changed from initial state"

        # Format execution history
        executed_commands = str(self.execution_history) if self.execution_history else "None"

        # Estimate progress based on execution results and scene state
        estimated_progress = 50  # Default
        if execution_success and execution_result and not (
                hasattr(execution_result, 'replan') and execution_result.replan):
            estimated_progress = 75
        elif not execution_success:
            estimated_progress = 25

        # Gather success/failure indicators
        success_indicators = []
        failure_indicators = []

        if execution_success:
            success_indicators.append("Execution completed without errors")
        if current_scene and len(current_scene) > 0:
            success_indicators.append("Objects detected in scene")
        if corrections_applied:
            success_indicators.append("Auto-correction applied successfully")

        if not execution_success:
            failure_indicators.append("Execution failed")
        if execution_result and hasattr(execution_result, 'replan') and execution_result.replan:
            failure_indicators.append("Replan explicitly requested")

        return PipelineState(
            original_task=original_task,
            current_iteration=planning_iteration,
            total_execution_time=total_execution_time,

            past_reasoning=plan_reason,
            generated_minispec=minispec_code,
            planning_time=planning_time,

            executed_commands=executed_commands,
            execution_result=str(execution_result) if execution_result else "None",
            execution_time=execution_time,
            execution_success=execution_success,

            initial_scene=self.initial_scene or "Not recorded",
            current_scene=current_scene,
            scene_changes=scene_changes,

            corrections_applied=corrections_applied,
            correction_details="Auto-correction applied" if corrections_applied else "No corrections needed",

            estimated_progress=estimated_progress,
            success_indicators="; ".join(success_indicators) if success_indicators else "None",
            failure_indicators="; ".join(failure_indicators) if failure_indicators else "None"
        )

    # ========== MESSAGING AND CONTROL ==========
    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def stop_controller(self):
        print_t("[C] Stopping controller...")
        self.controller_active = False

        # Signal cancellation to current interpreter
        self.cancellation_event.set()

        # Cancel current interpreter if it exists
        if self.current_interpreter is not None:
            print_t("[C] Cancelling current interpreter...")
            self.current_interpreter.cancel()

        # If drone has a stop method, call it
        if hasattr(self.drone, 'stop_all_commands'):
            print_t("[C] Stopping all drone commands...")
            self.drone.stop_all_commands()

        # Clean up YoloE client
        if hasattr(self, 'yoloe_client'):
            self.yoloe_client.close()

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            self.vision.update()
            YoloClient.plot_results_oi(image, self.vision.object_list)
        return image

    def get_latest_frame_path(self) -> Optional[str]:
        """
        Get the latest frame as a saved image file for VLM usage
        Returns the path to the saved image file

        INTEGRATION NOTE: This method is critical for VLM functionality.
        It saves the current frame to a temporary file that can be passed to the VLM.
        Files are saved to self.cache_folder and should be cleaned up periodically.
        """
        try:
            # Check for cancellation before processing
            if self.cancellation_event.is_set():
                print_t("[C] Frame capture cancelled")
                return None

            # Get image from shared frame
            image = self.shared_frame.get_image()
            if image is not None:
                # Generate unique filename
                img_path = os.path.join(self.cache_folder, f"latest_frame_{uuid.uuid4()}.jpg")

                # Handle both PIL Image and numpy array formats
                if hasattr(image, 'save'):
                    # PIL Image
                    image.save(img_path)
                else:
                    # numpy array - convert to PIL first
                    pil_image = Image.fromarray(image)
                    pil_image.save(img_path)

                print_t(f"[C] Latest frame saved to: {img_path}")
                return img_path
            else:
                print_t("[C] No image available in shared frame")
                return None
        except Exception as e:
            print_t(f"[C] Error saving latest frame: {e}")
            return None

    def execute_minispec(self, minispec: str):
        print_t(f"[C] Executing MiniSpec: {minispec}")

        # Create interpreter with separate message queue to avoid duplication
        # We'll use None to prevent the interpreter from sending code chunks
        interpreter = MiniSpecInterpreter(message_queue=None)
        self.current_interpreter = interpreter

        # Clear cancellation event for new execution
        self.cancellation_event.clear()

        interpreter.execute(minispec)
        self.execution_history = interpreter.execution_history
        ret_val = interpreter.ret_queue.get()
        print_t(f"[C] MiniSpec execution result: {ret_val}")

        self.current_interpreter = None
        return ret_val

    def execute_task_description_with_vlm(self, task_description: str, image_path: Optional[str] = None,
                                          use_vlm: bool = None):
        """
        Execute task description using the simplified enhanced agentic pipeline
        Now with simple VLM on/off toggle and consistent image provision, PLUS combined model support

        INTEGRATION NOTES:
        - Can use either dual-model approach (reasoning + writing) OR single combined model
        - VLM gets fresh current frame on every planning iteration when enabled (dual-model only)
        - Provided image_path is used only on first iteration (if available)
        - All subsequent iterations use get_latest_frame_path() for current visual context
        - Combined model approach bypasses VLM reasoning stage entirely
        - Simple toggle: VLM is either ON (with fallback) or OFF (LLM only)
        - FIXED: Always perform replan assessment after every execution (only once per iteration)

        Args:
            task_description: Natural language task description
            image_path: Optional path to image for VLM reasoning (used on first iteration only)
            use_vlm: Override VLM usage (None=use global setting, True=force on, False=force off)
        """
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return

        # Determine VLM usage - SIMPLE LOGIC (only applies to dual-model mode)
        if use_vlm is not None:
            vlm_enabled_for_task = use_vlm
            print_t(f"[C] VLM usage overridden: {'ON' if use_vlm else 'OFF'}")
        else:
            vlm_enabled_for_task = self.is_vlm_reasoning_enabled()
            print_t(f"[C] VLM usage (global setting): {'ON' if vlm_enabled_for_task else 'OFF'}")

        # Check if we're using combined model mode
        combined_mode = self.is_combined_mode_enabled()
        planning_approach = "COMBINED MODEL" if combined_mode else "DUAL-MODEL"

        print_t("[C] ========== ENHANCED AGENTIC TASK EXECUTION START ==========")
        print_t(f"[C] Task: {task_description}")
        print_t(f"[C] Image provided: {image_path if image_path else 'None'}")
        print_t(f"[C] Planning approach: {planning_approach}")
        if not combined_mode:
            print_t(f"[C] VLM will be used: {vlm_enabled_for_task}")
        print_t(f"[C] Manual stop available at any time")

        self.append_message('[TASK]: ' + task_description)
        if image_path:
            self.append_message(f'[IMAGE]: {image_path}')

        # Initialize enhanced pipeline state tracking
        self.task_start_time = time.time()
        self.initial_scene = self.vision.get_obj_list()
        self.pipeline_iterations = []
        self.replan_controller = SimplifiedEnhancedReplanController(self.planner.replan_llm)

        # Save current VLM state and set for this task - FIXED: Save both flags
        original_vlm_enabled = self.planner.vlm_enabled
        original_use_vlm_for_reasoning = self.planner.use_vlm_for_reasoning

        if use_vlm is not None and not combined_mode:
            # Override both flags for this task (only if not using combined mode)
            self.planner.vlm_enabled = use_vlm
            self.planner.use_vlm_for_reasoning = use_vlm
            print_t(f"[C] VLM state overridden for this task: {use_vlm}")

        ret_val = None
        planning_iteration = 0
        original_task = task_description

        try:
            while True:
                # Check if controller was stopped (for cancellation)
                if not self.controller_active:
                    print_t("[C] Task execution cancelled")
                    self.append_message("[CANCELLED] Task execution stopped")
                    break

                planning_iteration += 1
                print_t(f"[C] ===== PLANNING ITERATION {planning_iteration} =====")

                try:
                    # Get dual-stage planner response with optional VLM OR combined model
                    print_t(f"[C] Calling {planning_approach} planner...")
                    execution_start_time = time.time()

                    # Get current frame path for VLM if VLM is enabled AND we're using dual-model
                    current_image_path = None
                    if vlm_enabled_for_task and not combined_mode:
                        if planning_iteration == 1 and image_path:
                            # Use provided image on first iteration if available
                            current_image_path = image_path
                            print_t(f"[C] Using provided image for VLM: {image_path}")
                        else:
                            # Get current frame for all iterations when VLM is enabled
                            current_image_path = self.get_latest_frame_path()
                            if current_image_path:
                                print_t(f"[C] Using current frame for VLM: {current_image_path}")
                            else:
                                print_t(f"[C] Warning: Could not get current frame for VLM")
                    elif combined_mode:
                        print_t(f"[C] Combined model mode - no VLM reasoning")
                    else:
                        print_t(f"[C] VLM disabled - using LLM only")

                    # Call planner (automatically routes to combined or dual-model approach)
                    plan_reason, minispec_code = self.planner.plan(
                        task_description,
                        execution_history=self.execution_history,
                        image_path=current_image_path
                    )

                    # Check for cancellation after planning
                    if not self.controller_active:
                        print_t("[C] Task cancelled during planning")
                        break

                    planning_time = time.time() - execution_start_time
                    print_t(f"[C] Planning completed in {planning_time:.2f}s")

                    # Store the MiniSpec code for execution
                    self.current_plan = minispec_code

                    # Set target object for auto-correction based on plan
                    target_object = self._extract_target_objects_from_plan(minispec_code)
                    if target_object:
                        self.auto_corrector.set_target_object(target_object)

                    # Display the planning results with enhanced debugging
                    print_t(f"[C] ===== PLANNING RESULTS =====")
                    print_t(f"[C] Reasoning: {plan_reason}")
                    print_t(f"[C] Generated code: {minispec_code}")
                    print_t(f"[C] Code length: {len(minispec_code)} characters")
                    print_t(f"[C] Planning mode: {planning_approach}")
                    if current_image_path and not combined_mode:
                        print_t(f"[C] VLM used image: {current_image_path}")

                    # Send messages to UI/queue - Only send once here
                    if plan_reason:
                        self.append_message(f'[Plan]: {plan_reason}')
                    else:
                        self.append_message(f'[Plan]: No reasoning provided')

                    # Only send the code once, not during interpreter parsing
                    self.append_message(f'[Code]: {minispec_code}')

                    # Validate the generated code before execution
                    if not minispec_code or minispec_code.strip() == "":
                        print_t("[C] ERROR: Empty code generated")
                        self.append_message("[ERROR] Empty code generated - task failed")
                        break

                    # Check for cancellation before execution
                    if not self.controller_active:
                        print_t("[C] Task cancelled before execution")
                        break

                    # Execute the MiniSpec code
                    print_t("[C] ===== CODE EXECUTION START =====")
                    execution_start_time = time.time()
                    execution_success = True

                    try:
                        self.execution_time = time.time()
                        ret_val = self.execute_minispec(self.current_plan)

                        execution_time = time.time() - execution_start_time
                        print_t(f"[C] Code execution completed in {execution_time:.2f}s")
                        print_t(f"[C] Execution result: {ret_val}")

                    except Exception as e:
                        print_t(f"[C] ERROR during code execution: {e}")
                        self.append_message(f"[ERROR] Execution failed: {e}")
                        execution_success = False
                        execution_time = time.time() - execution_start_time
                        ret_val = None

                    # Apply auto-correction during planning downtime
                    print_t("[C] ===== AUTO-CORRECTION PHASE =====")
                    corrections_applied = self._apply_auto_correction_if_needed()

                except Exception as e:
                    print_t(f"[C] ERROR during planning: {e}")
                    self.append_message(f"[ERROR] Planning failed: {e}")
                    execution_success = False
                    planning_time = 0
                    execution_time = 0
                    corrections_applied = False
                    plan_reason = f"Planning failed: {e}"
                    minispec_code = "log('Planning failed')"

                # Check for cancellation after execution
                if not self.controller_active:
                    print_t("[C] Task cancelled after execution")
                    break

                # FIXED: Always perform replan assessment after every execution (only once per iteration)
                print_t(f"[C] ===== AUTOMATIC REPLAN ASSESSMENT STAGE =====")
                print_t(f"[C] Assessment triggered automatically after execution")

                # Create comprehensive pipeline state
                pipeline_state = self._create_pipeline_state(
                    planning_iteration=planning_iteration,
                    original_task=original_task,
                    plan_reason=plan_reason,
                    minispec_code=minispec_code,
                    planning_time=planning_time,
                    execution_result=ret_val,
                    execution_time=execution_time,
                    execution_success=execution_success,
                    corrections_applied=corrections_applied
                )

                # Get simplified replan assessment (only COMPLETE or CONTINUE)
                try:
                    replan_response = self.replan_controller.assess_pipeline_state(pipeline_state)

                    print_t(f"[C] Simplified Replan Assessment:")
                    print_t(f"[C] - Decision: {replan_response.decision.value}")
                    print_t(f"[C] - Confidence: {replan_response.confidence:.2f}")
                    print_t(f"[C] - Progress: {replan_response.progress_estimate}%")
                    print_t(f"[C] - Reasoning: {replan_response.reasoning}")

                    # Display assessment results
                    self.append_message(f"[AUTO-ASSESS] {replan_response.decision.value} - "
                                        f"Progress: {replan_response.progress_estimate}% - "
                                        f"Confidence: {replan_response.confidence:.2f}")
                    self.append_message(f"[REASONING] {replan_response.reasoning}")

                    if replan_response.safety_notes:
                        self.append_message(f"[SAFETY] {replan_response.safety_notes}")

                    # Handle simplified decision types
                    if replan_response.decision == ReplanDecision.COMPLETE:
                        print_t("[C] ===== TASK MARKED COMPLETE BY SIMPLIFIED CONTROLLER =====")
                        self.append_message("[COMPLETE] Task completed by simplified replan controller")
                        break

                    elif replan_response.decision == ReplanDecision.REPLAN_CONTINUE:
                        # Check if we should use the planner's replan assessment for natural language restart
                        print_t("[C] ===== CHECKING FOR NATURAL LANGUAGE REPLAN =====")

                        try:
                            # Use planner's assess_replan to get natural language task
                            decision, progress, feedback, new_task = self.planner.assess_replan(
                                original_task=original_task,
                                past_reasoning=plan_reason,
                                executed_commands=str(self.execution_history) if self.execution_history else "None",
                                current_scene=self.vision.get_obj_list()
                            )

                            if decision == "REPLAN" and new_task and new_task.strip():
                                print_t("[C] ===== RESTARTING WITH NEW NATURAL LANGUAGE TASK =====")
                                print_t(f"[C] Original task: {original_task}")
                                print_t(f"[C] New task from replan model: {new_task}")

                                # Update task description and restart planning loop
                                task_description = new_task
                                self.append_message(f"[RESTART] Replan model generated new task: {new_task}")

                                # Clear execution history for fresh start
                                self.execution_history = None
                                continue

                            else:
                                print_t("[C] ===== CONTINUING WITH CURRENT APPROACH =====")
                                if replan_response.next_guidance:
                                    print_t(f"[C] Guidance: {replan_response.next_guidance}")
                                    self.append_message(f"[CONTINUE] {replan_response.next_guidance}")
                                else:
                                    self.append_message(f"[CONTINUE] {replan_response.reasoning}")
                                continue

                        except Exception as replan_error:
                            print_t(f"[C] Error in natural language replan: {replan_error}")
                            # Fall back to continuing
                            print_t("[C] ===== FALLBACK: CONTINUING WITH CURRENT APPROACH =====")
                            if replan_response.next_guidance:
                                self.append_message(f"[CONTINUE] {replan_response.next_guidance}")
                            else:
                                self.append_message(f"[CONTINUE] {replan_response.reasoning}")
                            continue

                except Exception as e:
                    print_t(f"[C] ERROR during simplified replan assessment: {e}")
                    self.append_message(f"[ERROR] Simplified replan assessment failed: {e}")
                    # Fall back to completion for safety
                    print_t(f"[C] ===== SAFETY FALLBACK TO COMPLETION =====")
                    self.append_message("[SAFETY] Marking task complete due to assessment error")
                    break

        finally:
            # INTEGRATION FIX: Restore both VLM flags if they were overridden
            if use_vlm is not None and not combined_mode:
                self.planner.vlm_enabled = original_vlm_enabled
                self.planner.use_vlm_for_reasoning = original_use_vlm_for_reasoning
                print_t(
                    f"[C] VLM state restored: vlm_enabled={original_vlm_enabled}, use_vlm_for_reasoning={original_use_vlm_for_reasoning}")

            # Clear auto-correction target
            self.auto_corrector.clear_target()

        # Task completion or cancellation cleanup
        total_iterations = planning_iteration
        total_time = time.time() - self.task_start_time if self.task_start_time else 0

        if self.controller_active:
            print_t(f"[C] ========== ENHANCED TASK EXECUTION COMPLETE ==========")
            print_t(f"[C] Planning approach used: {planning_approach}")
            print_t(f"[C] Total planning iterations: {total_iterations}")
            print_t(f"[C] Total execution time: {total_time:.1f}s")
            self.append_message(
                f'\n[Task ended - {planning_approach} - {total_iterations} iterations, {total_time:.1f}s]')
        else:
            print_t(f"[C] ========== ENHANCED TASK EXECUTION CANCELLED ==========")
            print_t(f"[C] Completed iterations before cancellation: {total_iterations}")
            self.append_message(f'\n[Task cancelled - {total_iterations} iterations]')

        self.append_message('end')

        # Cleanup
        self.current_plan = None
        self.execution_history = None
        self.current_interpreter = None
        self.task_start_time = None
        self.initial_scene = None

    def execute_task_description(self, task_description: str):
        """
        Original method maintained for backward compatibility
        Calls the new enhanced method without image and uses global VLM setting

        INTEGRATION NOTE: This maintains existing behavior for code that doesn't use VLM
        """
        return self.execute_task_description_with_vlm(task_description, image_path=None, use_vlm=None)

    def execute_task_with_image(self, task_description: str, image_path: str):
        """
        Convenience method for executing tasks with image input
        Automatically enables VLM for this execution (only if using dual-model approach)

        INTEGRATION NOTE: This is a clean way to use VLM with a specific image
        """
        return self.execute_task_description_with_vlm(task_description, image_path=image_path, use_vlm=True)

    # ========== ROBOT CONTROL ==========
    def start_robot(self):
        print_t("[C] Connecting to robot...")
        self.drone.connect()
        print_t("[C] Starting robot...")
        self.drone.takeoff()
        self.drone.move_up(25)
        print_t("[C] Starting stream...")
        self.drone.start_stream()
        self.controller_wait_takeoff = False
        print_t("[C] Robot ready for tasks")

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        self.drone.stop_stream()
        self.controller_wait_takeoff = True
        print_t("[C] Robot stopped")

    def capture_loop(self, asyncio_loop):
        print_t("[C] Start capture loop...")

        # Wait for stream to be ready with timeout
        max_wait_time = 10  # seconds
        wait_start = time.time()
        frame_reader = None

        while frame_reader is None and (time.time() - wait_start) < max_wait_time:
            frame_reader = self.drone.get_frame_reader()
            if frame_reader is None:
                print_t("[C] Waiting for video stream to start...")
                time.sleep(0.5)

        if frame_reader is None:
            print_t("[C] ERROR: Could not get frame reader - video stream failed to start")
            return

        print_t("[C] Video stream ready, starting capture loop")

        while self.controller_active:
            try:
                self.drone.keep_active()

                # Get current frame with error handling
                if frame_reader and hasattr(frame_reader, 'frame'):
                    current_frame = frame_reader.frame
                    if current_frame is not None:
                        self.latest_frame = current_frame

                        frame = Frame(current_frame,
                                      frame_reader.depth if hasattr(frame_reader, 'depth') else None)

                        if self.yolo_client.is_local_service():
                            self.yolo_client.detect_local(frame)
                        else:
                            # asynchronously send image to yolo server
                            asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(frame))
                    else:
                        print_t("[C] Warning: Received None frame from stream")
                else:
                    print_t("[C] Warning: Frame reader lost, attempting to reconnect...")
                    frame_reader = self.drone.get_frame_reader()

            except Exception as e:
                print_t(f"[C] Error in capture loop: {e}")
                # Try to recover by getting a new frame reader
                try:
                    frame_reader = self.drone.get_frame_reader()
                except:
                    print_t("[C] Could not recover frame reader")

            time.sleep(0.10)

        print_t("[C] Capture loop stopping...")

        # Cancel all running tasks (if any)
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()

        # Clean shutdown
        try:
            self.drone.stop_stream()
            self.drone.land()
        except Exception as e:
            print_t(f"[C] Error during capture loop cleanup: {e}")

        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")