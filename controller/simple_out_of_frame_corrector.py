from typing import Optional, Tuple
import time
from enum import Enum

from .vision_skill_wrapper import ObjectInfo, VisionSkillWrapper
from .utils import print_t


class CorrectionType(Enum):
    """Simple correction types"""
    NONE = "none"
    TURN_LEFT = "turn_left" 
    TURN_RIGHT = "turn_right"
    BACKUP = "backup"
    APPROACH = "approach"


class SimpleOutOfFrameCorrector:
    """
    Simple post-skill correction system - like bowling bumpers
    Only activates AFTER complete skill execution, never during
    """
    
    def __init__(self, vision_skill: VisionSkillWrapper):
        self.vision_skill = vision_skill
        self.last_target_object = None
        self.enabled = True
        
        # Simple thresholds for corrections
        self.edge_threshold = 0.15      # Consider out-of-frame if within 15% of edge
        self.size_too_small = 0.05      # Object too far if width/height < 5%
        self.size_too_large = 0.6       # Object too close if width/height > 60%
        
        # Correction amounts (conservative)
        self.turn_correction = 15       # Turn 20 degrees for centering
        self.move_correction = 30       # Move 60cm for distance correction
        
        print_t("[BUMPER] Simple out-of-frame corrector initialized (bowling bumpers mode)")

    def set_target_object(self, object_name: str):
        """Set the object to watch for corrections after skill execution"""
        base_name = object_name.split('[')[0] if '[' in object_name else object_name
        self.last_target_object = base_name
        print_t(f"[BUMPER] Watching for '{base_name}' corrections")

    def check_and_correct(self, drone_wrapper) -> bool:
        """
        Check if correction is needed and apply simple correction
        Called AFTER skill execution completes, during planning downtime
        
        Args:
            drone_wrapper: Robot wrapper for movement commands
            
        Returns:
            True if correction was applied, False if no correction needed
        """
        if not self.enabled or not self.last_target_object:
            return False
        
        print_t(f"[BUMPER] === POST-SKILL CORRECTION CHECK ===")
        print_t(f"[BUMPER] Checking object: {self.last_target_object}")
        
        # Get current object state
        self.vision_skill.update()
        obj_info = self.vision_skill.get_obj_info(self.last_target_object)
        
        if obj_info is None:
            print_t(f"[BUMPER] Object '{self.last_target_object}' not visible - no correction possible")
            return False
        
        # Determine if correction is needed
        correction_type = self._analyze_object_position(obj_info)
        
        if correction_type == CorrectionType.NONE:
            print_t(f"[BUMPER] Object well-positioned - no correction needed")
            return False
        
        # Apply simple correction
        return self._apply_correction(correction_type, drone_wrapper, obj_info)

    def _analyze_object_position(self, obj_info: ObjectInfo) -> CorrectionType:
        """
        Analyze object position and determine correction type
        Simple logic - like bumpers in bowling
        
        Args:
            obj_info: Current object information
            
        Returns:
            Type of correction needed
        """
        x, y, w, h = obj_info.x, obj_info.y, obj_info.w, obj_info.h
        
        print_t(f"[BUMPER] Object position: x={x:.2f}, y={y:.2f}, size={w:.2f}x{h:.2f}")
        
        # Check if object is too far left or right
        if x < self.edge_threshold:
            print_t(f"[BUMPER] Object too far left (x={x:.2f})")
            return CorrectionType.TURN_LEFT
        elif x > (1.0 - self.edge_threshold):
            print_t(f"[BUMPER] Object too far right (x={x:.2f})")
            return CorrectionType.TURN_RIGHT
        
        # Check if object is too close or far (based on size)
        avg_size = (w + h) / 2
        if avg_size > self.size_too_large:
            print_t(f"[BUMPER] Object too close (size={avg_size:.2f})")
            return CorrectionType.BACKUP
        elif avg_size < self.size_too_small:
            print_t(f"[BUMPER] Object too far (size={avg_size:.2f})")
            return CorrectionType.APPROACH
        
        # Object is well-positioned
        return CorrectionType.NONE

    def _apply_correction(self, correction_type: CorrectionType, drone_wrapper, obj_info: ObjectInfo) -> bool:
        """
        Apply simple correction command
        
        Args:
            correction_type: Type of correction to apply
            drone_wrapper: Robot wrapper for movement
            obj_info: Object information for fine-tuning
            
        Returns:
            True if correction was applied successfully
        """
        print_t(f"[BUMPER] === APPLYING CORRECTION: {correction_type.value.upper()} ===")
        
        try:
            if correction_type == CorrectionType.TURN_LEFT:
                print_t(f"[BUMPER] Turning left {self.turn_correction} degrees to center object")
                drone_wrapper.turn_ccw(self.turn_correction)
                
            elif correction_type == CorrectionType.TURN_RIGHT:
                print_t(f"[BUMPER] Turning right {self.turn_correction} degrees to center object")
                drone_wrapper.turn_cw(self.turn_correction)
                
            elif correction_type == CorrectionType.BACKUP:
                print_t(f"[BUMPER] Backing up {self.move_correction}cm - object too close")
                drone_wrapper.move_backward(self.move_correction)
                
            elif correction_type == CorrectionType.APPROACH:
                print_t(f"[BUMPER] Moving forward {self.move_correction}cm - object too far")
                drone_wrapper.move_forward(self.move_correction)
            
            # Brief pause after correction
            time.sleep(0.5)
            print_t(f"[BUMPER] Correction applied successfully")
            return True
            
        except Exception as e:
            print_t(f"[BUMPER] Error applying correction: {e}")
            return False

    def enable(self):
        """Enable auto-correction"""
        self.enabled = True
        print_t("[BUMPER] Auto-correction enabled")

    def disable(self):
        """Disable auto-correction"""
        self.enabled = False
        print_t("[BUMPER] Auto-correction disabled")

    def set_thresholds(self, edge_threshold: float = None, size_small: float = None, 
                      size_large: float = None):
        """
        Adjust correction thresholds
        
        Args:
            edge_threshold: Distance from edge to trigger centering (0.0-0.5)
            size_small: Size threshold for "too far" detection
            size_large: Size threshold for "too close" detection
        """
        if edge_threshold is not None:
            self.edge_threshold = max(0.05, min(0.4, edge_threshold))
        if size_small is not None:
            self.size_too_small = max(0.01, min(0.2, size_small))
        if size_large is not None:
            self.size_too_large = max(0.3, min(0.8, size_large))
            
        print_t(f"[BUMPER] Thresholds updated - edge:{self.edge_threshold:.2f}, "
                f"small:{self.size_too_small:.2f}, large:{self.size_too_large:.2f}")

    def set_correction_amounts(self, turn_degrees: int = None, move_distance: int = None):
        """
        Adjust correction movement amounts
        
        Args:
            turn_degrees: Degrees to turn for centering corrections
            move_distance: Distance (cm) to move for approach/backup corrections
        """
        if turn_degrees is not None:
            self.turn_correction = max(10, min(45, turn_degrees))
        if move_distance is not None:
            self.move_correction = max(30, min(120, move_distance))
            
        print_t(f"[BUMPER] Correction amounts updated - turn:{self.turn_correction}°, "
                f"move:{self.move_correction}cm")

    def clear_target(self):
        """Clear the current target object"""
        self.last_target_object = None
        print_t("[BUMPER] Target object cleared")