# %%
import pygame
import numpy as np
import time

# Initialize Pygame
pygame.init()

# Grid size
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 50, 50
SIZE = WIDTH // ROWS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Initialize Grid
grid = np.zeros((ROWS, COLS))
#  grid = np.random.choice([0, 1], ROWS*COLS, p=[0.8, 0.2]).reshape(ROWS, COLS)

# Initialize Window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Conway's Game of Life")

# Start & Pause Button
start_button = pygame.Rect(WIDTH - 150, 50, 100, 50)
pause_button = pygame.Rect(WIDTH - 150, 150, 100, 50)

running = False

# Function to draw the grid
def draw(window, grid):
    window.fill(BLACK)
    for i in range(ROWS):
        for j in range(COLS):
            color = WHITE if grid[i][j] else BLACK
            pygame.draw.rect(window, color, (i * SIZE, j * SIZE, SIZE, SIZE), 0)

    # Draws p
    pygame.draw.rect(window, GREEN, start_button, 2)
    pygame.draw.rect(window, RED, pause_button, 2)
    pygame.display.update()

# Function to apply the rules
def apply_rules(grid):
    new_grid = grid.copy()
    for i in range(ROWS):
        for j in range(COLS):
            state = grid[i][j]
            neighbors = int((grid[i - 1, j - 1] + grid[i - 1, j] + grid[i - 1, (j + 1) % COLS] +
                             grid[i, j - 1] + grid[i, (j + 1) % COLS] +
                             grid[(i + 1) % ROWS, j - 1] + grid[(i + 1) % ROWS, j] + grid[(i + 1) % ROWS, (j + 1) % COLS]) - state)
            if state == 1 and (neighbors < 2 or neighbors > 3):
                new_grid[i, j] = 0
            elif state == 0 and neighbors == 3:
                new_grid[i, j] = 1
    return new_grid

# Main Loop
while True:
    draw(window, grid)
    if running:
        time.sleep(0.1)
        grid = apply_rules(grid)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            row, col = x // SIZE, y // SIZE
            if start_button.collidepoint(x, y):
                running = True
            elif pause_button.collidepoint(x, y):
                running = False
            elif not running:
                grid[row, col] = not grid[row, col]
        if event.type == pygame.MOUSEMOTION and not running:
            x, y = pygame.mouse.get_pos()
            buttons = pygame.mouse.get_pressed()
            if buttons[0]:
                row, col = x // SIZE, y // SIZE
                grid[row, col] = 1

# %%
