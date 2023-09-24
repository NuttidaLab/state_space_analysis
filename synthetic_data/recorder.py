import numpy as np
import pygame
import math
import random
import pickle
import time

# Constants
CANVAS_SIZE = (1000, 1000)  # Adjust this to your canvas size
# NUM_TRIALS = 20
NUM_TRIALS = 5
NUM_TIMESTEPS = 1300

# Initialize Pygame
pygame.init()
canvas = pygame.display.set_mode(CANVAS_SIZE)
pygame.display.set_caption('Joystick Simulation')

# Function to calculate the angle between two points
def calculate_angle(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return 360 - (angle + 360 if angle < 0 else angle)

# Create arrays to store cursor and target data
cursor_data = np.zeros((NUM_TRIALS, NUM_TIMESTEPS, 3))  # Angle, X, Y
target_data = np.zeros((NUM_TRIALS, 3))  # Angle, X, Y

# Simulation loop
for trial in range(NUM_TRIALS):
    # Wait for left-click to start the trial
    clicked = False
    while not clicked:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = True

    # Center the mouse
    pygame.mouse.set_pos(CANVAS_SIZE[0] / 2, CANVAS_SIZE[1] / 2)

    # Calculate a random target angle and coordinates
    target_angle = random.uniform(0, 360)
    target_x = CANVAS_SIZE[0] / 2 + CANVAS_SIZE[0] * 0.4 * math.cos(math.radians(target_angle))
    target_y = CANVAS_SIZE[1] / 2 + CANVAS_SIZE[0] * 0.4 * math.sin(math.radians(target_angle))

    target_data[trial, 0] = target_angle
    target_data[trial, 1] = target_x
    target_data[trial, 2] = target_y

    for timestep in range(NUM_TIMESTEPS):
        # Handle mouse events to continuously update cursor position
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                cursor_angle = calculate_angle(CANVAS_SIZE[0] / 2, CANVAS_SIZE[1] / 2, x, y)
                cursor_data[trial, timestep, 0] = cursor_angle
                cursor_data[trial, timestep, 1] = x
                cursor_data[trial, timestep, 2] = y
            else:
                cursor_data[trial, timestep, 0] = cursor_data[trial, timestep - 1, 0]
                cursor_data[trial, timestep, 1] = cursor_data[trial, timestep - 1, 1]
                cursor_data[trial, timestep, 2] = cursor_data[trial, timestep - 1, 2]

        # Draw on the canvas
        canvas.fill((255, 255, 255))  # Clear canvas

        # Draw the valid canvas area
        pygame.draw.circle(canvas, (0, 0, 0), (CANVAS_SIZE[0] // 2, CANVAS_SIZE[1] // 2), CANVAS_SIZE[0] * 0.4, 1)

        # Draw the target
        pygame.draw.line(canvas, (255, 0, 0), (target_x - 20, target_y - 20), (target_x + 20, target_y + 20), 2)
        pygame.draw.line(canvas, (255, 0, 0), (target_x - 20, target_y + 20), (target_x + 20, target_y - 20), 2)

        # Draw the cursor
        pygame.draw.circle(canvas, (0, 0, 0), (int(x), int(y)), 5)

        # Display timestep and trial counter at the top right
        font = pygame.font.Font(None, 36)
        text = font.render(f"Ts: {timestep+1}/{NUM_TIMESTEPS} Tr: {trial+1}/{NUM_TRIALS}| X={target_x:.1f} Y={target_y:.1f} A={target_angle:.1f}| X={x:.1f} Y={y:.1f} A={cursor_angle:.1f}", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topright = (CANVAS_SIZE[0] - 10, 10)
        canvas.blit(text, text_rect)

        pygame.display.update()

# Quit Pygame
pygame.quit()

# save_data = [CANVAS_SIZE, NUM_TRIALS, NUM_TIMESTEPS, CANVAS_SIZE[0] * 0.4, cursor_data, target_data]
# pickle.dump(save_data, open("top_bias.pkl", "wb"))