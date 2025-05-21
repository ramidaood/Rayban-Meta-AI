import cv2
import numpy as np
import pygame
import os

class NavigationSystem:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Center zone is now 15% of screen width (reduced from 20%)
        self.center_zone_width = screen_width * 0.15
        self.center_point = screen_width / 2
        self.center_start = self.center_point - (self.center_zone_width / 2)
        self.center_end = self.center_point + (self.center_zone_width / 2)
        
        # Alert thresholds
        self.distance_threshold = 0.4  # Reduced from 0.5
        self.confidence_threshold = 0.5
        
        # Initialize audio
        self.initialize_audio()
        
    def initialize_audio(self):
        """Initialize pygame and audio system"""
        try:
            pygame.init()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.left_channel = pygame.mixer.Channel(0)
            self.right_channel = pygame.mixer.Channel(1)
            self.sound = self.load_sound()
            print("Navigation system initialized successfully")
        except Exception as e:
            print(f"Error initializing navigation system: {e}")
            self.sound = None
            
    def load_sound(self):
        """Load the sound effect"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sound_path = os.path.join(current_dir, 'SurroundTest', 'Assets', 'Audio', 'soundEffect.wav')
            
            if not os.path.exists(sound_path):
                print("Sound file not found!")
                return None
                
            return pygame.mixer.Sound(sound_path)
        except Exception as e:
            print(f"Error loading sound: {e}")
            return None
            
    def play_sound(self, side):
        """Play sound on specified side"""
        if not self.sound:
            return
            
        pygame.mixer.stop()
        if side == 'left':
            self.left_channel.set_volume(1.0, 0.0)
            self.left_channel.play(self.sound)
        else:
            self.right_channel.set_volume(0.0, 1.0)
            self.right_channel.play(self.sound)
            
    def analyze_objects(self, objects):
        """Analyze objects and determine navigation alerts"""
        if not objects:
            return None, None
            
        # Group objects by zone
        zones = {
            "left": [],
            "center": [],
            "right": []
        }
        
        # Sort objects into zones
        for obj in objects:
            x = obj['center_x']
            if self.center_start <= x < self.center_end:
                zones["center"].append(obj)
            elif x < self.center_start:
                zones["left"].append(obj)
            else:
                zones["right"].append(obj)
                
        # Check center zone first
        if zones["center"]:
            # Find closest object in center
            center_object = min(zones["center"], key=lambda x: x['distance'])
            
            # If center object is close enough and has good confidence
            if (center_object['distance'] < self.distance_threshold and 
                center_object['confidence'] > self.confidence_threshold):
                
                # Count potential blockers on each side
                left_blockers = len([obj for obj in zones["left"] 
                                   if obj['distance'] < self.distance_threshold * 1.2])
                right_blockers = len([obj for obj in zones["right"] 
                                    if obj['distance'] < self.distance_threshold * 1.2])
                
                # Determine which side has fewer obstacles
                if left_blockers < right_blockers:
                    return "left", center_object
                else:
                    return "right", center_object
                    
        return None, None
        
    def draw_debug_info(self, frame):
        """Draw debug information on frame"""
        # Draw center zone
        cv2.line(frame, 
                (int(self.center_start), 0),
                (int(self.center_start), frame.shape[0]),
                (0, 255, 0), 2)
        cv2.line(frame, 
                (int(self.center_end), 0),
                (int(self.center_end), frame.shape[0]),
                (0, 255, 0), 2)
                
        # Draw center point
        cv2.circle(frame, 
                  (int(self.center_point), frame.shape[0] // 2),
                  5, (0, 0, 255), -1)
                  
        return frame 