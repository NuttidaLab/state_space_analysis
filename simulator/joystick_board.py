import numpy as np
import pygame
import math

rng = np.random.default_rng()

class stickBoard:
    
    def __init__(self, CANVAS_SIZE = 1000, ACTIVE_AREA = 0.4) -> None:
        self.CANVAS_SIZE = CANVAS_SIZE
        self.ACTIVE_AREA = ACTIVE_AREA
        self.CENTER = CANVAS_SIZE // 2

    def calc_angle(self, x1, y1, x2, y2):
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return 360 - (angle + 360 if angle < 0 else angle)

    def calc_coord(self, angle, radius, center_x, center_y):
        x = center_x + radius * math.sin(math.radians(90 - angle))
        y = center_y - radius * math.sin(math.radians(angle))
        return x, y
    
    def save_data(self, file_name):
        np.savez_compressed(file_name, cursor_data=self.cursor_data, target_data=self.target_data)
    
    def setup_exp(self, N_SESS = 1, N_BLOCKS = 1, N_TS = 1000):
        self.cursor_data = np.zeros((N_SESS, N_BLOCKS*5, N_TS, 3))
        self.target_data = np.zeros((N_SESS, N_BLOCKS*5, 3))

        self.target_angles = np.array([[15, 51, 87, 123, 159]*N_BLOCKS]*N_SESS)
        self.target_angles = rng.permuted(self.target_angles, axis=1)

        self.N_TRIAL = N_BLOCKS * 5
        self.N_SESS = N_SESS
        self.N_TS = N_TS

    def setup_game(self):
        pygame.init()
        self.canvas = pygame.display.set_mode((self.CANVAS_SIZE, self.CANVAS_SIZE))
        pygame.display.set_caption('Joystick Simulation')

    def draw_board(self, sess_idx, trial_idx, timestep, cur_x, cur_y, cur_angle, tar_x, tar_y, tar_angle):
        canvas = self.canvas
        canvas.fill((255, 255, 255))  # Clear canvas

        for ts in self.cursor_data[sess_idx, trial_idx]:
            pygame.draw.circle(canvas, (200, 200, 200), (int(ts[1]), int(ts[2])), 5)

        # Draw the valid canvas area
        pygame.draw.circle(canvas, (0, 0, 0), (self.CENTER, self.CENTER), self.CANVAS_SIZE * self.ACTIVE_AREA, 5)

        # Draw the center
        pygame.draw.circle(canvas, (50, 50, 50), (self.CENTER, self.CENTER), 10)

        # Draw horizontal line at the center diameter
        pygame.draw.line(canvas, (50, 50, 50), (self.CENTER - self.CANVAS_SIZE * self.ACTIVE_AREA, self.CENTER), (self.CENTER + self.CANVAS_SIZE * self.ACTIVE_AREA, self.CENTER), 3)

        # Draw small circles at these points on the circumference 15, 51, 87, 123, 159 degrees
        for angle in [15, 51, 87, 123, 159]:
            x, y = self.calc_coord(angle, self.CANVAS_SIZE * self.ACTIVE_AREA, self.CENTER, self.CENTER)
            pygame.draw.circle(canvas, (50, 50, 50), (int(x), int(y)), 8)
            # label these circles offset by 20 pixels outside the circumfrance
            offset = 30
            font = pygame.font.Font(None, 36)
            text = font.render(f"{angle}", True, (50, 50, 50))
            text_rect = text.get_rect()
            text_rect.center = (int(x + offset * math.sin(math.radians(90 - angle))), int(y - offset * math.sin(math.radians(angle))))
            canvas.blit(text, text_rect)
        
        # Draw the target
        pygame.draw.line(canvas, (255, 0, 0), (tar_x - 20, tar_y - 20), (tar_x + 20, tar_y + 20), 8)
        pygame.draw.line(canvas, (255, 0, 0), (tar_x - 20, tar_y + 20), (tar_x + 20, tar_y - 20), 8)

        # Draw the cursor
        pygame.draw.circle(canvas, (50, 50, 50), (int(cur_x), int(cur_y)), 10)

        # Draw a dashed line from current cursor position to origin
        pygame.draw.line(canvas, (50, 50, 50), (self.CENTER, self.CENTER), (cur_x, cur_y), 3)

        # Draw a light red line from target to origin
        pygame.draw.line(canvas, (255, 0, 0), (self.CENTER, self.CENTER), (tar_x, tar_y), 3)

        # Display timestep and trial_idx counter at the top right
        font = pygame.font.Font(None, 36)
        text = font.render(f"Angle ={cur_angle:.0f} / {tar_angle:.0f} | X= {cur_x:.0f} / {tar_x:.0f} Y= {cur_y:.0f} / {tar_y:.0f} | Ts: {timestep+1}/{self.N_TS} Tr: {trial_idx+1}/{self.N_TRIAL}", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topright = (self.CANVAS_SIZE - 10, 10)
        canvas.blit(text, text_rect)

        pygame.display.update()

    def run_trial(self, sess_idx, trial_idx):

        # Wait for left-click to start the trial
        clicked = False
        while not clicked:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked = True

        # Center the mouse
        pygame.mouse.set_pos(self.CENTER, self.CENTER)
        
        target_angle = self.target_angles[sess_idx, trial_idx]
        target_x, target_y = self.calc_coord(target_angle, self.CANVAS_SIZE * self.ACTIVE_AREA, self.CENTER, self.CENTER)
        self.target_data[sess_idx, trial_idx, 0] = target_angle
        self.target_data[sess_idx, trial_idx, 1] = target_x
        self.target_data[sess_idx, trial_idx, 2] = target_y

        for timestep in range(self.N_TS):
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    x, y = event.pos
                    cursor_angle = self.calc_angle(self.CENTER, self.CENTER, x, y)

                    # If its a mouse motion and the movement is outside the active area:
                    if (x - self.CENTER) ** 2 + (y - self.CENTER) ** 2 > (self.CANVAS_SIZE * self.ACTIVE_AREA) ** 2:
                        # Calculate the new x and y coordinates
                        x, y = self.calc_coord(cursor_angle, self.CANVAS_SIZE * self.ACTIVE_AREA, self.CENTER, self.CENTER)
                    
                    self.cursor_data[sess_idx, trial_idx, timestep, :] = [cursor_angle, x , y]
                    
                else:
                    self.cursor_data[sess_idx, trial_idx, timestep, :] = self.cursor_data[sess_idx, trial_idx, timestep - 1, :]

            self.draw_board(sess_idx, trial_idx, timestep, x, y, cursor_angle, target_x, target_y, target_angle)

    def stop_game(self):
        pygame.quit()