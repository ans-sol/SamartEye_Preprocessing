import carla
import pygame
import json
import os
import sys

# Initialize CARLA client and world
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town02')  # Explicitly load the town
except RuntimeError as e:
    print(f"Failed to connect to CARLA: {e}")
    sys.exit(1)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Camera Position Picker - Town01")
font = pygame.font.SysFont('Arial', 24)

# Position storage - now stores full transform
camera_transforms = []
filename = "camera_transforms.json"

def save_transforms():
    with open(filename, 'w') as f:
        json.dump([{
            "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
            "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll}
        } for t in camera_transforms], f, indent=2)
    print(f"Saved {len(camera_transforms)} transforms to {filename}")

def load_transforms():
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return [carla.Transform(
                carla.Location(x=t["location"]["x"], y=t["location"]["y"], z=t["location"]["z"]),
                carla.Rotation(pitch=t["rotation"]["pitch"], yaw=t["rotation"]["yaw"], roll=t["rotation"]["roll"])
            ) for t in data]
    return []

# Main game loop
def main():
    global camera_transforms
    camera_transforms = load_transforms()
    clock = pygame.time.Clock()
    running = True
    spectator = world.get_spectator()

    def render_ui():
        """Render UI with town information and controls"""
        screen.fill((0, 0, 40))  # Dark blue background
        
        # Show town information
        map_name = world.get_map().name.split('/')[-1]
        town_text = font.render(f"Current Town: {map_name}", True, (255, 255, 0))
        screen.blit(town_text, (10, 10))
        
        # Render control instructions
        controls = [
            "LEFT CLICK: Save current transform",
            "RIGHT CLICK: Remove last transform",
            "MIDDLE MOUSE: Print current transform",
            "R: Reset rotation to (pitch=0, yaw=0)",
            "S: Save to file",
            "ESC: Quit"
        ]
        
        for i, text in enumerate(controls):
            text_surface = font.render(text, True, (200, 200, 255))
            screen.blit(text_surface, (10, 50 + i * 30))
        
        # Display transform information
        transform = spectator.get_transform()
        transform_text = [
            f"Current Transform:",
            f"Position:",
            f"  X: {transform.location.x:.2f}",
            f"  Y: {transform.location.y:.2f}",
            f"  Z: {transform.location.z:.2f}",
            f"Rotation:",
            f"  Pitch: {transform.rotation.pitch:.1f}",
            f"  Yaw: {transform.rotation.yaw:.1f}",
            f"  Roll: {transform.rotation.roll:.1f}"
        ]
        
        for i, text in enumerate(transform_text):
            text_surface = font.render(text, True, (255, 200, 200))
            screen.blit(text_surface, (400, 50 + i * 25))
        
        # Show saved transforms count
        count_text = font.render(f"Saved Transforms: {len(camera_transforms)}", True, (200, 255, 200))
        screen.blit(count_text, (10, 250))
        
        pygame.display.flip()

    print("\n=== CARLA Camera Transform Picker ===")
    print(f"Connected to {world.get_map().name}")
    print("Controls:")
    print("- LEFT CLICK: Save current spectator transform")
    print("- RIGHT CLICK: Remove last saved transform")
    print("- MIDDLE MOUSE: Print current transform")
    print("- R: Reset rotation to (pitch=0, yaw=0)")
    print("- S: Save to file")
    print("- ESC: Quit\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                transform = spectator.get_transform()
                
                if event.button == 1:  # Left click
                    camera_transforms.append(transform)
                    print(f"Added transform {len(camera_transforms)}:")
                    print(f"  Position: {transform.location}")
                    print(f"  Rotation: {transform.rotation}")
                    
                elif event.button == 3:  # Right click
                    if camera_transforms:
                        removed = camera_transforms.pop()
                        print(f"Removed transform: {removed}")
                        
                elif event.button == 2:  # Middle mouse
                    print(f"Current transform: {transform}")
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_transforms()
                elif event.key == pygame.K_r:  # Reset rotation
                    current_loc = spectator.get_transform().location
                    spectator.set_transform(carla.Transform(
                        current_loc,
                        carla.Rotation(pitch=0, yaw=0, roll=0)
                    ))
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        render_ui()
        clock.tick(60)

    save_transforms()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()