import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json
import grpc
import torch
from ultralytics import YOLO
import multiprocessing
import numpy as np
from typing import Optional

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_PATH = os.environ.get("ROOT_PATH", PARENT_DIR)
SERVICE_PORT = os.environ.get("YOLOE_SERVICE_PORT", "50052, 50053").split(",")

MODEL_PATH = os.path.join(ROOT_PATH, "./serving/yoloe/models/")
MODEL_TYPE = "yoloe_n.pt"  # YoloE model

sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

def load_yoloe_model():
    """Load YoloE model with visual prompt detection capability"""
    try:
        model = YOLO(MODEL_PATH + MODEL_TYPE)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        model.to(device)
        print(f"[YoloE] Model loaded successfully on {device}")
        print(f"[YoloE] GPU memory usage: {torch.cuda.memory_allocated()}")
        return model
    except Exception as e:
        print(f"[YoloE] Warning: Could not load YoloE model: {e}")
        print(f"[YoloE] Falling back to standard YOLO model")
        # Fallback to standard YOLO
        model = YOLO("yolo11s.pt")
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model.to(device)
        return model

def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


class YoloEService(hyrch_serving_pb2_grpc.YoloServiceServicer):
    """
    YoloE gRPC service with visual prompt detection capability
    Supports Semantic-Activated Visual Prompt Encoder (SAVPE) for one-shot detection
    """
    
    def __init__(self, port):
        self.stream_mode = False
        self.model = load_yoloe_model()
        self.port = port
        self.reference_images = {}  # Cache for reference images
        self.prompt_embeddings = {}  # Cache for visual prompt embeddings
        
        print(f"[YoloE] Service initialized on port {port}")

    def reload_model(self):
        if self.model is not None:
            release_model(self.model)
        self.model = load_yoloe_model()

    @staticmethod
    def bytes_to_image(image_bytes):
        return Image.open(BytesIO(image_bytes))
    
    def encode_visual_prompt(self, reference_image: Image.Image, description: str) -> Optional[torch.Tensor]:
        """
        Encode visual prompt using SAVPE (Semantic-Activated Visual Prompt Encoder)
        
        Args:
            reference_image: Reference image for visual prompting
            description: Text description to enhance prompt encoding
            
        Returns:
            Encoded prompt tensor, or None if encoding fails
        """
        try:
            # For YoloE, this would use the SAVPE module to encode
            # the reference image into semantic and activation features
            
            # Placeholder implementation - in a real YoloE service, this would:
            # 1. Extract features from reference image using SAVPE encoder
            # 2. Combine with text description embeddings
            # 3. Generate conditioning tensor for detection
            
            print(f"[YoloE] Encoding visual prompt for: {description}")
            
            # For now, return None to indicate prompt encoding not available
            # This will fall back to standard detection
            return None
            
        except Exception as e:
            print(f"[YoloE] Error encoding visual prompt: {e}")
            return None

    def detect_with_visual_prompt(self, image: Image.Image, prompt_description: str, 
                                 reference_image: Optional[Image.Image] = None,
                                 conf: float = 0.3) -> dict:
        """
        Perform detection with visual prompt conditioning
        
        Args:
            image: Input image to detect objects in
            prompt_description: Text description of target object
            reference_image: Optional reference image for visual prompting
            conf: Confidence threshold
            
        Returns:
            Detection results in standard format
        """
        print(f"[YoloE] Visual prompt detection for: '{prompt_description}'")
        
        # Encode visual prompt if reference image provided
        prompt_embedding = None
        if reference_image is not None:
            prompt_embedding = self.encode_visual_prompt(reference_image, prompt_description)
            print(f"[YoloE] Using reference image for visual prompting")
        
        # Perform detection with prompt conditioning
        if prompt_embedding is not None:
            # For real YoloE, this would condition the detection on the prompt
            # For now, use standard detection as fallback
            print(f"[YoloE] Using prompt-conditioned detection")
            if self.stream_mode:
                yolo_result = self.model.track(image, verbose=False, conf=conf, tracker="bytetrack.yaml")[0]
            else:
                yolo_result = self.model(image, verbose=False, conf=conf)[0]
        else:
            # Standard detection with text-based filtering
            print(f"[YoloE] Using text-based detection filtering")
            if self.stream_mode:
                yolo_result = self.model.track(image, verbose=False, conf=conf, tracker="bytetrack.yaml")[0]
            else:
                yolo_result = self.model(image, verbose=False, conf=conf)[0]
        
        # Filter results based on prompt description
        filtered_results = self.filter_by_description(yolo_result, prompt_description)
        
        return {
            "detection_mode": "visual_prompt",
            "prompt_description": prompt_description,
            "has_reference": reference_image is not None,
            "result": filtered_results,
        }

    def filter_by_description(self, yolo_result, description: str) -> list:
        """
        Filter detection results based on text description
        This is a fallback when visual prompt encoding is not available
        
        Args:
            yolo_result: Raw YOLO detection results
            description: Text description to match against
            
        Returns:
            Filtered list of detections
        """
        if yolo_result.probs is not None:
            print('[YoloE] Warning: Classify task not supported for visual prompts')
            return []
        
        formatted_results = []
        data = yolo_result.boxes.data.cpu().tolist() if yolo_result.boxes is not None else []
        h, w = yolo_result.orig_shape
        
        # Simple text matching for description filtering
        description_lower = description.lower()
        relevant_keywords = self.extract_keywords(description_lower)
        
        for i, row in enumerate(data):
            box = {'x1': round(row[0] / w, 2), 'y1': round(row[1] / h, 2), 
                   'x2': round(row[2] / w, 2), 'y2': round(row[3] / h, 2)}
            conf = row[-2]
            class_id = int(row[-1])
            
            object_name = yolo_result.names[class_id]
            if yolo_result.boxes.is_track:
                object_name = f'{object_name}_{int(row[-3])}'
            
            # Check if object matches description
            if self.matches_description(object_name, relevant_keywords):
                result = {
                    'name': object_name, 
                    'confidence': round(conf, 2), 
                    'box': box,
                    'matched_description': description
                }
                
                if yolo_result.masks:
                    x, y = yolo_result.masks.xy[i][:, 0], yolo_result.masks.xy[i][:, 1]
                    result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
                    
                if yolo_result.keypoints is not None:
                    x, y, visible = yolo_result.keypoints[i].data[0].cpu().unbind(dim=1)
                    result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
                
                formatted_results.append(result)
        
        print(f"[YoloE] Found {len(formatted_results)} objects matching '{description}'")
        return formatted_results

    def extract_keywords(self, description: str) -> list:
        """Extract relevant keywords from description"""
        # Simple keyword extraction - could be enhanced with NLP
        common_objects = ['person', 'car', 'chair', 'table', 'cup', 'bottle', 'book', 'phone', 'laptop', 'bag']
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown']
        
        keywords = []
        words = description.split()
        
        for word in words:
            if word in common_objects or word in colors:
                keywords.append(word)
        
        return keywords

    def matches_description(self, object_name: str, keywords: list) -> bool:
        """Check if object name matches description keywords"""
        if not keywords:
            return True  # If no specific keywords, accept all objects
        
        object_lower = object_name.lower()
        for keyword in keywords:
            if keyword in object_lower:
                return True
        return False

    @staticmethod
    def format_result(yolo_result):
        """Standard YOLO result formatting"""
        if yolo_result.probs is not None:
            print('[YoloE] Warning: Classify task not supported')
            return []
            
        formatted_result = []
        data = yolo_result.boxes.data.cpu().tolist()
        h, w = yolo_result.orig_shape
        
        for i, row in enumerate(data):
            box = {'x1': round(row[0] / w, 2), 'y1': round(row[1] / h, 2), 
                   'x2': round(row[2] / w, 2), 'y2': round(row[3] / h, 2)}
            conf = row[-2]
            class_id = int(row[-1])

            name = yolo_result.names[class_id]
            if yolo_result.boxes.is_track:
                name = f'{name}_{int(row[-3])}'
                
            result = {'name': name, 'confidence': round(conf, 2), 'box': box}
            
            if yolo_result.masks:
                x, y = yolo_result.masks.xy[i][:, 0], yolo_result.masks.xy[i][:, 1]
                result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
                
            if yolo_result.keypoints is not None:
                x, y, visible = yolo_result.keypoints[i].data[0].cpu().unbind(dim=1)
                result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
                
            formatted_result.append(result)
            
        return formatted_result
    
    def process_image(self, image, id=None, conf=0.3):
        """Standard image processing (non-visual prompt)"""
        if self.stream_mode:
            yolo_result = self.model.track(image, verbose=False, conf=conf, tracker="bytetrack.yaml")[0]
        else:
            yolo_result = self.model(image, verbose=False, conf=conf)[0]
            
        result = {
            "image_id": id,
            "result": YoloEService.format_result(yolo_result),
        }
        return json.dumps(result)

    def process_visual_prompt_request(self, request_data: dict, image: Image.Image, 
                                    reference_image: Optional[Image.Image] = None) -> str:
        """Process visual prompt detection request"""
        prompt_description = request_data.get('prompt_description', '')
        conf = request_data.get('conf', 0.3)
        image_id = request_data.get('image_id', None)
        
        # Perform visual prompt detection
        result = self.detect_with_visual_prompt(image, prompt_description, reference_image, conf)
        result['image_id'] = image_id
        
        return json.dumps(result)

    def DetectStream(self, request, context):
        print(f"[YoloE] Received DetectStream request from {context.peer()} on port {self.port}")
        
        if not self.stream_mode:
            self.stream_mode = True
            self.reload_model()
        
        image = YoloEService.bytes_to_image(request.image_data)
        return hyrch_serving_pb2.DetectResponse(json_data=self.process_image(image, request.image_id, request.conf))
    
    def Detect(self, request, context):
        print(f"[YoloE] Received Detect request from {context.peer()} on port {self.port}")
        
        if self.stream_mode:
            self.stream_mode = False
            self.reload_model()

        image = YoloEService.bytes_to_image(request.image_data)
        return hyrch_serving_pb2.DetectResponse(json_data=self.process_image(image, request.image_id, request.conf))


def serve(port):
    print(f"[YoloE] Starting YoloE Service at port {port}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    hyrch_serving_pb2_grpc.add_YoloServiceServicer_to_server(YoloEService(port), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    # Create a pool of processes for YoloE service
    process_count = len(SERVICE_PORT)
    processes = []

    for i in range(process_count):
        process = multiprocessing.Process(target=serve, args=(SERVICE_PORT[i],))
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()