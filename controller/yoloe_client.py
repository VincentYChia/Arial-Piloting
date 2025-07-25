from io import BytesIO
from PIL import Image
from typing import Optional, List, Tuple
import json, sys, os
import queue
import grpc
import asyncio
import base64
from numpy.typing import NDArray
import numpy as np

from .yolo_client import SharedFrame, Frame
from .utils import print_t

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(PARENT_DIR, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
YOLOE_SERVICE_PORT = os.environ.get("YOLOE_SERVICE_PORT", "50053")


class YoloEClient():
    """
    YoloE client for visual prompt detection using gRPC
    Adapted from the existing YoloGRPCClient pattern
    """

    def __init__(self, shared_frame: SharedFrame = None):
        # Initialize gRPC channel and stub (like YoloGRPCClient)
        self.channel = grpc.insecure_channel(f'{VISION_SERVICE_IP}:{YOLOE_SERVICE_PORT}')
        self.stub = hyrch_serving_pb2_grpc.YoloServiceStub(self.channel)

        # Async initialization
        self.is_async_inited = False
        self.channel_async = None
        self.stub_async = None

        # Standard settings (copied from YoloGRPCClient)
        self.image_size = (640, 352)
        self.frame_queue = queue.Queue()
        self.shared_frame = shared_frame
        self.frame_id_lock = asyncio.Lock()
        self.frame_id = 0

        # YoloE specific configurations
        self.confidence_threshold = 0.3
        self.reference_image_cache = {}

        print_t(f"[YE] YoloE gRPC client initialized to {VISION_SERVICE_IP}:{YOLOE_SERVICE_PORT}")

    def init_async_channel(self):
        """Initialize async gRPC channel (copied from YoloGRPCClient)"""
        if not self.is_async_inited:
            self.channel_async = grpc.aio.insecure_channel(f'{VISION_SERVICE_IP}:{YOLOE_SERVICE_PORT}')
            self.stub_async = hyrch_serving_pb2_grpc.YoloServiceStub(self.channel_async)
            self.is_async_inited = True

    def is_local_service(self):
        return VISION_SERVICE_IP == 'localhost'

    @staticmethod
    def image_to_bytes(image):
        """Convert PIL image to bytes (copied from YoloGRPCClient)"""
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    def detect_local(self, frame: Frame, description: str, conf: float = 0.3) -> Optional[dict]:
        """
        Local gRPC detection for YoloE (adapted from YoloGRPCClient.detect_local)
        """
        print_t(f"[YE] Local gRPC detection for: {description}")

        image = frame.image
        image_bytes = YoloEClient.image_to_bytes(image.resize(self.image_size))
        self.frame_queue.put(frame)

        # Create gRPC request (using same format as YOLO but could be extended for YoloE)
        # Note: This assumes YoloE service accepts the same protobuf format
        # You might need to modify this if YoloE has different message structure
        detect_request = hyrch_serving_pb2.DetectRequest(
            image_data=image_bytes,
            conf=conf
            # TODO: Add description field if YoloE protobuf supports it
            # description=description
        )

        try:
            response = self.stub.DetectStream(detect_request)
            json_results = json.loads(response.json_data)

            if self.shared_frame is not None:
                self.shared_frame.set(self.frame_queue.get(), json_results)

            return json_results

        except grpc.RpcError as e:
            print_t(f"[YE] gRPC error: {e}")
            return None

    async def detect_async(self, frame: Frame, description: str, conf: float = 0.3) -> Optional[dict]:
        """
        Async gRPC detection for YoloE (adapted from YoloGRPCClient.detect)
        """
        if not self.is_async_inited:
            self.init_async_channel()

        if self.is_local_service():
            return self.detect_local(frame, description, conf)

        image = frame.image
        image_bytes = YoloEClient.image_to_bytes(image)

        async with self.frame_id_lock:
            image_id = self.frame_id
            self.frame_queue.put((self.frame_id, frame))
            self.frame_id += 1

        detect_request = hyrch_serving_pb2.DetectRequest(
            image_id=image_id,
            image_data=image_bytes,
            conf=conf
        )

        try:
            response = await self.stub_async.Detect(detect_request)
            json_results = json.loads(response.json_data)

            if self.frame_queue.empty():
                return None

            # Discard old images (copied from YoloGRPCClient)
            while self.frame_queue.queue[0][0] < json_results['image_id']:
                self.frame_queue.get()
            # Discard old results
            if self.frame_queue.queue[0][0] > json_results['image_id']:
                return None

            if self.shared_frame is not None:
                self.shared_frame.set(self.frame_queue.get()[1], json_results)

            return json_results

        except grpc.RpcError as e:
            print_t(f"[YE] Async gRPC error: {e}")
            return None

    def detect_by_description(self, frame: Frame, description: str, conf: float = 0.3) -> Tuple[Optional[str], bool]:
        """
        Main interface for description-based detection using gRPC
        This replaces the HTTP-based approach

        Args:
            frame: Current camera frame
            description: Text description of object to find
            conf: Confidence threshold

        Returns:
            Tuple of (object_name_with_position, replan_needed)
        """
        print_t(f"[YE] === DESCRIPTION-BASED DETECTION ===")
        print_t(f"[YE] Looking for: {description}")
        print_t(f"[YE] Using gRPC detection")

        try:
            # Use local gRPC detection
            result = self.detect_local(frame, description, conf)

            if result is None:
                print_t(f"[YE] Detection service unavailable for: {description}")
                return None, True  # Replan needed due to service failure

            # Parse detection results (same format as before)
            detections = result.get('result', [])
            if not detections:
                print_t(f"[YE] No objects found matching: {description}")
                return None, False  # No replan needed, just no objects found

            # Find the best matching detection
            best_detection = max(detections, key=lambda x: x.get('confidence', 0))

            object_name = best_detection['name']
            box = best_detection['box']
            confidence = best_detection['confidence']

            # Calculate center x-coordinate
            x_center = (box['x1'] + box['x2']) / 2

            # Format result as expected by MiniSpec (object_name[x_value])
            result_string = f"{object_name}[{x_center:.2f}]"

            print_t(f"[YE] Detection successful: {result_string} (confidence: {confidence:.2f})")
            return result_string, False

        except Exception as e:
            print_t(f"[YE] Error in gRPC detection: {e}")
            return None, True  # Replan needed due to error

    def set_confidence_threshold(self, threshold: float):
        """Set default confidence threshold for detections"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        print_t(f"[YE] Confidence threshold set to: {self.confidence_threshold}")

    def clear_reference_cache(self):
        """Clear cached reference images"""
        self.reference_image_cache.clear()
        print_t("[YE] Reference image cache cleared")

    def close(self):
        """Clean up gRPC resources"""
        if self.channel:
            self.channel.close()
        if self.channel_async:
            asyncio.create_task(self.channel_async.close())
        print_t("[YE] YoloE gRPC client resources cleaned up")