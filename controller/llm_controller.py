from PIL import Image
import queue, time, os, json
from typing import Optional, Tuple
import asyncio
import uuid
from enum import Enum
import re

from .shared_frame import SharedFrame, Frame
from .yolo_client import YoloClient
from .yolo_grpc_client import YoloGRPCClient
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
        print_t("[C] Initializing LLMController with dual-model and probe support...")

        self.shared_frame = SharedFrame()
        if use_http:
            self.yolo_client = YoloClient(shared_frame=self.shared_frame)
        else:
            self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
        self.vision = VisionSkillWrapper(self.shared_frame)
        self.latest_frame = None
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
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

        # ========== PROBE FUNCTIONALITY RESTORED ==========
        self.low_level_skillset.add_skill(
            LowLevelSkillItem("probe", self.planner.probe, "Probe the LLM for reasoning",
                              args=[SkillArg("question", str)]))

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
        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset,
                          vision_skill=self.vision)

        self.current_plan = None
        self.execution_history = None
        self.execution_time = time.time()

        print_t("[C] LLMController with dual-model and probe support initialization complete")

    # ========== MODEL CONFIGURATION METHODS ==========
    def set_reasoning_model(self, model_name: str, temperature: float = None):
        """Configure the reasoning model"""
        print_t(f"[C] Setting reasoning model to: {model_name}")
        self.planner.set_reasoning_model(model_name, temperature)

    def set_minispec_model(self, model_name: str, temperature: float = None):
        """Configure the MiniSpec generation model"""
        print_t(f"[C] Setting MiniSpec model to: {model_name}")
        self.planner.set_minispec_model(model_name, temperature)

    # ========== SKILL METHODS ==========
    def skill_time(self) -> Tuple[float, bool]:
        return time.time() - self.execution_time, False

    def skill_goto(self, object_name: str) -> Tuple[None, bool]:
        print_t(f'[C] Executing goto: {object_name}')
        if '[' in object_name:
            x = float(object_name.split('[')[1].split(']')[0])
        else:
            x = self.vision.object_x(object_name)[0]

        print_t(f'[C] GOTO target x-coordinate: {x} (type: {type(x)})')

        if x > 0.55:
            turn_amount = int((x - 0.5) * 70)
            print_t(f'[C] Turning clockwise {turn_amount} degrees')
            self.drone.turn_cw(turn_amount)
        elif x < 0.45:
            turn_amount = int((0.5 - x) * 70)
            print_t(f'[C] Turning counter-clockwise {turn_amount} degrees')
            self.drone.turn_ccw(turn_amount)

        print_t('[C] Moving forward 110 units')
        self.drone.move_forward(110)
        return None, False

    def skill_take_picture(self) -> Tuple[None, bool]:
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
        time.sleep(s)
        return None, False

    # ========== MESSAGING AND CONTROL ==========
    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def stop_controller(self):
        print_t("[C] Stopping controller...")
        self.controller_active = False

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            self.vision.update()
            YoloClient.plot_results_oi(image, self.vision.object_list)
        return image

    def execute_minispec(self, minispec: str):
        print_t(f"[C] Executing MiniSpec: {minispec}")
        interpreter = MiniSpecInterpreter(self.message_queue)
        interpreter.execute(minispec)
        self.execution_history = interpreter.execution_history
        ret_val = interpreter.ret_queue.get()
        print_t(f"[C] MiniSpec execution result: {ret_val}")
        return ret_val

    def execute_task_description(self, task_description: str):
        """
        Execute task description using the new dual-model planning system
        Enhanced with cancellation support for concurrent execution
        """
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return

        print_t("[C] ========== TASK EXECUTION START ==========")
        print_t(f"[C] Task: {task_description}")
        self.append_message('[TASK]: ' + task_description)

        ret_val = None
        planning_iteration = 0

        while True:
            # Check if controller was stopped (for cancellation)
            if not self.controller_active:
                print_t("[C] Task execution cancelled")
                self.append_message("[CANCELLED] Task execution stopped")
                break

            planning_iteration += 1
            print_t(f"[C] ===== PLANNING ITERATION {planning_iteration} =====")

            try:
                # Get dual-stage planner response
                print_t("[C] Calling dual-stage planner...")
                execution_start_time = time.time()

                plan_reason, minispec_code = self.planner.plan(
                    task_description,
                    execution_history=self.execution_history
                )

                # Check for cancellation after planning
                if not self.controller_active:
                    print_t("[C] Task cancelled during planning")
                    break

                planning_time = time.time() - execution_start_time
                print_t(f"[C] Planning completed in {planning_time:.2f}s")

                # Store the MiniSpec code for execution
                self.current_plan = minispec_code

                # Display the planning results with enhanced debugging
                print_t(f"[C] ===== PLANNING RESULTS =====")
                print_t(f"[C] Reasoning: {plan_reason}")
                print_t(f"[C] Generated code: {minispec_code}")
                print_t(f"[C] Code length: {len(minispec_code)} characters")

                # Send messages to UI/queue
                if plan_reason:
                    self.append_message(f'[Plan]: {plan_reason}')
                else:
                    self.append_message(f'[Plan]: No reasoning provided')

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

                try:
                    self.execution_time = time.time()
                    ret_val = self.execute_minispec(self.current_plan)

                    execution_time = time.time() - execution_start_time
                    print_t(f"[C] Code execution completed in {execution_time:.2f}s")
                    print_t(f"[C] Execution result: {ret_val}")

                except Exception as e:
                    print_t(f"[C] ERROR during code execution: {e}")
                    self.append_message(f"[ERROR] Execution failed: {e}")
                    break

            except Exception as e:
                print_t(f"[C] ERROR during planning: {e}")
                self.append_message(f"[ERROR] Planning failed: {e}")
                break

            # Check for cancellation after execution
            if not self.controller_active:
                print_t("[C] Task cancelled after execution")
                break

            # Handle replanning logic
            if ret_val and hasattr(ret_val, 'replan') and ret_val.replan:
                print_t(f"[C] ===== REPLANNING REQUESTED =====")
                print_t(f"[C] Replan reason: {ret_val.value}")
                self.append_message(f"[REPLAN] {ret_val.value}")
                continue
            else:
                print_t("[C] ===== TASK COMPLETED =====")
                break

        # Task completion or cancellation cleanup
        total_iterations = planning_iteration
        if self.controller_active:
            print_t(f"[C] ========== TASK EXECUTION COMPLETE ==========")
            print_t(f"[C] Total planning iterations: {total_iterations}")
            self.append_message(f'\n[Task ended]')
        else:
            print_t(f"[C] ========== TASK EXECUTION CANCELLED ==========")
            print_t(f"[C] Completed iterations before cancellation: {total_iterations}")
            self.append_message(f'\n[Task cancelled]')

        self.append_message('end')

        # Cleanup
        self.current_plan = None
        self.execution_history = None

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
        frame_reader = self.drone.get_frame_reader()
        while self.controller_active:
            self.drone.keep_active()
            self.latest_frame = frame_reader.frame
            frame = Frame(frame_reader.frame,
                          frame_reader.depth if hasattr(frame_reader, 'depth') else None)

            if self.yolo_client.is_local_service():
                self.yolo_client.detect_local(frame)
            else:
                # asynchronously send image to yolo server
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(frame))
            time.sleep(0.10)
        # Cancel all running tasks (if any)
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        self.drone.stop_stream()
        self.drone.land()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")