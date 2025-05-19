import pygame
import os
import sys

def initialize_audio():
    """Initialize pygame and audio system"""
    try:
        pygame.init()
        print("Pygame initialized successfully")
        
        # Initialize the mixer with specific parameters
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        print("Audio mixer initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing audio: {e}")
        return False

def load_sounds():
    """Load the sound effects"""
    try:
        # Get the absolute path to the sound file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sound_path = os.path.join(current_dir, 'SurroundTest', 'Assets', 'Audio', 'soundEffect.wav')
        
        print(f"Attempting to load sound from: {sound_path}")
        print(f"File exists: {os.path.exists(sound_path)}")
        
        if not os.path.exists(sound_path):
            print("Sound file not found!")
            return None
            
        sound = pygame.mixer.Sound(sound_path)
        print("Sound loaded successfully")
        return sound
    except Exception as e:
        print(f"Error loading sound: {e}")
        return None

def play_sound(sound, channel, side):
    """Play sound on specified channel with stereo effect"""
    try:
        if sound:
            if side == 'left':
                # Full volume on left, no volume on right
                channel.set_volume(1.0, 0.0)
            else:  # right
                # No volume on left, full volume on right
                channel.set_volume(0.0, 1.0)
            channel.play(sound)
            print(f"Playing sound on {side} side")
        else:
            print("No sound loaded to play")
    except Exception as e:
        print(f"Error playing sound: {e}")

def main():
    print("Starting Sound Test Program...")
    
    # Initialize audio
    if not initialize_audio():
        print("Failed to initialize audio system")
        sys.exit(1)
    
    # Create channels for left and right
    try:
        left_channel = pygame.mixer.Channel(0)
        right_channel = pygame.mixer.Channel(1)
        print("Audio channels created successfully")
    except Exception as e:
        print(f"Error creating audio channels: {e}")
        sys.exit(1)
    
    # Load sound
    sound = load_sounds()
    if not sound:
        print("Failed to load sound file")
        sys.exit(1)
    
    print("\nSound Test Program Ready!")
    print("Press 'L' for left sound")
    print("Press 'R' for right sound")
    print("Press 'Q' to quit")
    
    # Create a small window (required for Pygame)
    screen = pygame.display.set_mode((300, 200))
    pygame.display.set_caption("Sound Test")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    print("Playing left sound...")
                    play_sound(sound, left_channel, 'left')
                elif event.key == pygame.K_r:
                    print("Playing right sound...")
                    play_sound(sound, right_channel, 'right')
                elif event.key == pygame.K_q:
                    running = False
    
    pygame.quit()
    print("Program ended")

if __name__ == "__main__":
    main() 