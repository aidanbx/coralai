import pygame
from Hexagon import *

COLOR_OPTIONS = [OFF_WHITE, OBSTACLE_BLACK, FOOD_GREEN, CHEMOATTRACTANT_YELLOW, POISON_PURPLE, CHEMOREPELLENT_RED]

HEXAGON_SIZE = 20 # Distance from center of hexagon to edge? Probably
GRID_WIDTH = 11
GRID_HEIGHT = 11

pygame.init()
info = pygame.display.Info()
# SCREEN_WIDTH = info.current_w
# SCREEN_HEIGHT = info.current_h - 55
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

COLOR_BAR_WIDTH = 50
COLOR_BAR_HEIGHT = SCREEN_HEIGHT

# Define a list of colors to choose from
selected_color = None
color_buttons = [
    pygame.Rect(SCREEN_WIDTH - COLOR_BAR_WIDTH, 
                i * COLOR_BAR_HEIGHT // (len(COLOR_OPTIONS)),
                COLOR_BAR_WIDTH,
                COLOR_BAR_HEIGHT // len(COLOR_OPTIONS))
    for i in range(len(COLOR_OPTIONS))
]

def update_hexagon_state(hexagon):
    # Update the state of a hexagon in pygame
    if selected_color is not None and hexagon is not None:
        hexagon.color = selected_color

def update_screen(screen, grid):
    screen.fill(DARK_GREY)  # Clear the screen with dark_grey color
    
    for i, button in enumerate(color_buttons): 
        if COLOR_OPTIONS[i] == selected_color:
            pygame.draw.rect(screen, selected_color, button)
            pygame.draw.rect(screen, OFF_WHITE, button, 3)
        else:
            pygame.draw.rect(screen, COLOR_OPTIONS[i], button)
    
    # Draw the hexagons
    for row in grid.grid:
        for hexagon in row:
            vertices = hexagon.get_vertices(HEXAGON_SIZE, GRID_WIDTH, GRID_HEIGHT)
            shifted_vertices = [(x + SCREEN_WIDTH / 2, y + SCREEN_HEIGHT / 2) for x, y in vertices]
            pygame.draw.polygon(screen, hexagon.color, shifted_vertices, 0)
            pygame.draw.polygon(screen, DARK_GREY, shifted_vertices, 1)

            # Label the hexagon with its x, y value
            label_font = pygame.font.SysFont(None, 12)
            label_text = f"{hexagon.x_index}, {hexagon.y_index}"
            label_surface = label_font.render(label_text, True, POISON_PURPLE)
            label_rect = label_surface.get_rect(center=(shifted_vertices[0][0] - 20, shifted_vertices[0][1]))
            screen.blit(label_surface, label_rect)
    
    # Draw the current mouse position in the top left corner (x, y)
    mouse_pos = pygame.mouse.get_pos()
    shifted_mouse_pos = (mouse_pos[0], mouse_pos[1])
    mouse_pos_text = f"Mouse Position: {shifted_mouse_pos[0]}, {shifted_mouse_pos[1]}"
    mouse_pos_font = pygame.font.SysFont(None, 20)
    mouse_pos_surface = mouse_pos_font.render(mouse_pos_text, True, POISON_PURPLE)
    mouse_pos_rect = mouse_pos_surface.get_rect(topright=(300, 5))
    screen.blit(mouse_pos_surface, mouse_pos_rect)

    return screen


def hexagon_clicked(mouse_pos, grid):
    # Convert pygame coordinates to hexagon in HexagonGrid
    offset_x = mouse_pos[0] - (SCREEN_WIDTH / 2)
    offset_y = mouse_pos[1] - (SCREEN_HEIGHT / 2)
    col = int(offset_x / (3/2 * HEXAGON_SIZE))
    row = int((offset_y - (col % 2) * HEXAGON_SIZE / 2) / HEXAGON_SIZE)

    print(offset_x, offset_x, col, row)

    if 0 <= col < grid.width and 0 <= row < grid.height:
        return grid.grid[col][row]
    return None
    

def main():
    global selected_color
    grid = HexagonGrid(GRID_WIDTH, GRID_HEIGHT)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos() # WAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAh not 
                if mouse_pos[0] > SCREEN_WIDTH - COLOR_BAR_WIDTH:
                    for i, button in enumerate(color_buttons):
                        if button.collidepoint(mouse_pos):
                            selected_color = COLOR_OPTIONS[i]
                else:
                    clicked_hexagon = hexagon_clicked(mouse_pos, grid)
                    update_hexagon_state(clicked_hexagon)
    
        screen = update_screen(screen, grid)
        pygame.display.flip()
        clock.tick(90) # FPS
    
    pygame.quit()

if __name__ == "__main__":
    main()

