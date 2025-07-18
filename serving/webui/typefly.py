import queue
import sys, os
import asyncio
import io, time
import gradio as gr
from flask import Flask, Response
from threading import Thread, Lock
import argparse
import threading

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(PARENT_DIR)
from controller.llm_controller import LLMController
from controller.utils import print_t
from controller.llm_wrapper import GPT4, LLAMA3, TINY_LLAMA
from controller.abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TypeFly:
    def __init__(self, robot_type, use_http=False):
        # create a cache folder
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.message_queue = queue.Queue()
        self.message_queue.put(self.cache_folder)
        self.llm_controller = LLMController(robot_type, use_http, self.message_queue)
        self.system_stop = False
        self.ui = gr.Blocks(title="TypeFly")
        self.asyncio_loop = asyncio.get_event_loop()
        self.use_llama3 = False

        # ENHANCED CONCURRENT EXECUTION PROTECTION
        self.task_lock = Lock()
        self.current_task_thread = None
        self.task_cancelled = False
        self.task_in_progress = False

        default_sentences = [
            "Find something I can eat.",
            "What you can see?",
            "Follow that ball for 20 seconds",
            "Find a chair for me.",
            "Go to the chair without book.",
        ]
        with self.ui:
            gr.HTML(open(os.path.join(CURRENT_DIR, 'header.html'), 'r').read())
            gr.HTML(open(os.path.join(CURRENT_DIR, 'drone-pov.html'), 'r').read())
            gr.ChatInterface(self.process_message, fill_height=False,
                             examples=default_sentences).queue()

    def checkbox_llama3(self):
        """Fixed for dual-model architecture"""
        self.use_llama3 = not self.use_llama3
        if self.use_llama3:
            print_t(f"[S] Switching to LLAMA3 for both reasoning and minispec models")
            self.llm_controller.set_reasoning_model(LLAMA3)
            self.llm_controller.set_minispec_model(LLAMA3)
        else:
            print_t(f"[S] Switching to TINY_LLAMA for both reasoning and minispec models")
            self.llm_controller.set_reasoning_model(TINY_LLAMA)
            self.llm_controller.set_minispec_model(TINY_LLAMA)

    def cancel_current_task(self):
        """Enhanced cancellation method"""
        print_t("[S] ========== CANCELLATION INITIATED ==========")
        self.task_cancelled = True

        # Stop the controller (this will set the cancellation event)
        self.llm_controller.stop_controller()

        # Wait for current task to finish with timeout
        if self.current_task_thread and self.current_task_thread.is_alive():
            print_t("[S] Waiting for task thread to terminate...")
            self.current_task_thread.join(timeout=5.0)
            if self.current_task_thread.is_alive():
                print_t("[S] Warning: Task thread did not terminate cleanly")
            else:
                print_t("[S] Task thread terminated successfully")

        # Clear the message queue to prevent stale messages
        print_t("[S] Clearing message queue...")
        cleared_count = 0
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        print_t(f"[S] Cleared {cleared_count} messages from queue")

        # Reset controller state
        self.llm_controller.controller_active = True
        self.llm_controller.cancellation_event.clear()
        self.task_cancelled = False
        self.task_in_progress = False

        print_t("[S] ========== CANCELLATION COMPLETE ==========")

    def execute_task_safely(self, message):
        """Execute task with enhanced error handling and cancellation support"""
        try:
            print_t(f"[S] ========== TASK THREAD START ==========")
            print_t(f"[S] Starting task execution: {message}")
            self.task_in_progress = True

            # Call the controller's execute method
            self.llm_controller.execute_task_description(message)

            print_t(f"[S] Task execution completed: {message}")
            print_t(f"[S] ========== TASK THREAD END ==========")

        except Exception as e:
            print_t(f"[S] Task execution failed: {e}")
            self.message_queue.put(f"[ERROR] Task failed: {e}")
            self.message_queue.put('end')
        finally:
            self.task_in_progress = False

    def process_message(self, message, history):
        print_t(f"[S] ========== MESSAGE PROCESSING START ==========")
        print_t(f"[S] Receiving task description: {message}")

        if message == "exit":
            self.llm_controller.stop_controller()
            self.system_stop = True
            yield "Shutting down..."
            return
        elif len(message) == 0:
            yield "[WARNING] Empty command!"
            return

        # ENHANCED CONCURRENT EXECUTION HANDLING
        with self.task_lock:
            # Cancel any running task
            if self.current_task_thread and self.current_task_thread.is_alive():
                print_t("[S] ⚠️ New task received while previous task running")
                yield "⚠️ Cancelling previous task..."

                # Cancel the previous task
                self.cancel_current_task()
                yield "✅ Previous task cancelled. Starting new task..."

                # Small delay to ensure cleanup is complete
                time.sleep(0.5)

            # Start new task
            print_t(f"[S] Starting new task thread for: {message}")
            self.current_task_thread = Thread(target=self.execute_task_safely, args=(message,))
            self.current_task_thread.daemon = True
            self.current_task_thread.start()

        # ENHANCED MESSAGE PROCESSING
        complete_response = ''
        start_time = time.time()
        last_activity_time = time.time()
        message_count = 0

        while True:
            try:
                msg = self.message_queue.get(timeout=2.0)
                last_activity_time = time.time()
                message_count += 1

            except queue.Empty:
                # Check if thread is still alive
                if not self.current_task_thread.is_alive():
                    print_t("[S] Task thread terminated without sending 'end' message")
                    if self.task_in_progress:
                        yield complete_response + "\n⚠️ Task ended unexpectedly"
                    else:
                        yield complete_response + "\n✅ Task completed"
                    return

                # Check for overall timeout (5 minutes)
                if time.time() - start_time > 300:
                    print_t("[S] Task timeout - cancelling")
                    self.cancel_current_task()
                    yield complete_response + "\n⚠️ Task timed out and was cancelled"
                    return

                # Check for inactivity timeout (30 seconds)
                if time.time() - last_activity_time > 30:
                    print_t("[S] Task inactivity timeout - checking status")
                    if not self.current_task_thread.is_alive():
                        print_t("[S] Task thread died during inactivity")
                        yield complete_response + "\n⚠️ Task stopped unexpectedly"
                        return

                continue

            # Handle different message types
            if isinstance(msg, tuple):
                # Image message
                print_t(f"[S] Received image message: {msg}")
                history.append((None, msg))

            elif isinstance(msg, str):
                if msg == 'end':
                    print_t(f"[S] Task completed successfully (processed {message_count} messages)")
                    yield complete_response + "\n✅ Task completed successfully!"
                    return

                # Process text messages
                print_t(f"[S] Processing message: {msg[:50]}...")

                # Handle streaming messages (those ending with \\)
                if msg.endswith('\\\\'):
                    complete_response += msg.rstrip('\\\\')
                else:
                    # Regular messages get a newline
                    if msg.startswith('[LOG]'):
                        complete_response += '\n' + msg
                    else:
                        complete_response += msg + '\n'

            # Yield the updated response
            yield complete_response

    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
            frame = self.llm_controller.get_latest_frame(True)
            if frame is None:
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        print_t("[S] Starting TypeFly system...")

        asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        asyncio_thread.daemon = True
        asyncio_thread.start()

        self.llm_controller.start_robot()
        llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,))
        llmc_thread.daemon = True
        llmc_thread.start()

        app = Flask(__name__)

        @app.route('/drone-pov/')
        def video_feed():
            return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

        flask_thread = Thread(target=app.run,
                              kwargs={'host': 'localhost', 'port': 50000, 'debug': False, 'use_reloader': False})
        flask_thread.daemon = True
        flask_thread.start()

        print_t("[S] Launching Gradio interface...")
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)

        try:
            while True:
                time.sleep(1)
                if self.system_stop:
                    break
        except KeyboardInterrupt:
            print_t("[S] Keyboard interrupt received")
            self.system_stop = True

        print_t("[S] Shutting down TypeFly...")

        # Cancel any running tasks
        if self.current_task_thread and self.current_task_thread.is_alive():
            print_t("[S] Cancelling running task before shutdown...")
            self.cancel_current_task()

        # Wait for threads to finish
        if self.current_task_thread and self.current_task_thread.is_alive():
            self.current_task_thread.join(timeout=5)

        self.llm_controller.stop_robot()

        # Clean cache folder
        try:
            for file in os.listdir(self.cache_folder):
                os.remove(os.path.join(self.cache_folder, file))
            print_t("[S] Cache folder cleaned")
        except Exception as e:
            print_t(f"[S] Error cleaning cache: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_virtual_robot', action='store_true')
    parser.add_argument('--use_http', action='store_true')
    parser.add_argument('--gear', action='store_true')

    args = parser.parse_args()
    robot_type = RobotType.TELLO
    if args.use_virtual_robot:
        robot_type = RobotType.VIRTUAL
    elif args.gear:
        robot_type = RobotType.GEAR

    print_t(f"[S] Starting TypeFly with robot type: {robot_type}")
    typefly = TypeFly(robot_type, use_http=args.use_http)
    typefly.run()