import pygame
from Hexagon import *

COLOR_OPTIONS = [OFF_WHITE, OBSTACLE_BLACK, FOOD_GREEN, CHEMOATTRACTANT_YELLOW, POISON_PURPLE, CHEMOREPELLENT_RED]

HEXAGON_SIZE = 20
GRID_WIDTH = 10
GRID_HEIGHT = 10
SCREEN_WIDTH = 200 + GRID_WIDTH * 40
SCREEN_HEIGHT = 200 + GRID_HEIGHT * 40
COLOR_BAR_WIDTH = 50
COLOR_BAR_HEIGHT = SCREEN_HEIGHT
COLOR_BAR_X = SCREEN_WIDTH - COLOR_BAR_WIDTH
COLOR_BAR_Y = 0
COLOR_BAR_SPACING = COLOR_BAR_HEIGHT // (len(COLOR_OPTIONS))  # Add 1 to the denominator to create spacing between the rectangles

# Define a list of colors to choose from
selected_color = None
color_buttons = [
    pygame.Rect(COLOR_BAR_X, COLOR_BAR_Y + i * COLOR_BAR_SPACING, COLOR_BAR_WIDTH, COLOR_BAR_HEIGHT // len(COLOR_OPTIONS))
    for i in range(len(COLOR_OPTIONS))
]

def update_hexagon_state(hexagon):
    # Update the state of a hexagon in pygame
    if selected_color is not None and hexagon is not None:
        hexagon.color = selected_color
        print(hexagon.q, hexagon.r)

def update_screen(screen, hexagons):
    screen.fill(DARK_GREY)  # Clear the screen with dark_grey color
    
    for i, button in enumerate(color_buttons): 
        if COLOR_OPTIONS[i] == selected_color:
            pygame.draw.rect(screen, selected_color, button)
            pygame.draw.rect(screen, OFF_WHITE, button, 3)
        else:
            pygame.draw.rect(screen, COLOR_OPTIONS[i], button)
    
    # Draw the hexagons
    for row in hexagons:
        for hexagon in row:
            vertices = hexagon.get_vertices(HEXAGON_SIZE)
            offset_to_center_x = (GRID_WIDTH * HEXAGON_SIZE)
            offset_to_center_y = (GRID_HEIGHT * HEXAGON_SIZE)
            shifted_vertices = [(x + SCREEN_WIDTH / 2 - offset_to_center_x, y + SCREEN_HEIGHT / 2 - offset_to_center_y) for x, y in vertices]
            pygame.draw.polygon(screen, hexagon.color, shifted_vertices, 0)
            pygame.draw.polygon(screen, DARK_GREY, shifted_vertices, 1)
            # Label the hexagon with its x, y value
            label_font = pygame.font.SysFont(None, 12)
            label_text = f"{hexagon.x}, {hexagon.y}"
            label_surface = label_font.render(label_text, True, POISON_PURPLE)
            label_rect = label_surface.get_rect(center=(shifted_vertices[0][0] - 5, shifted_vertices[0][1]))
            screen.blit(label_surface, label_rect)
        
    return screen


def hexagon_clicked(mouse_pos, hexagons):
    # Convert pygame coordinates to hexagon in HexagonGrid
    offset_to_center_x = (GRID_WIDTH * HEXAGON_SIZE)
    offset_to_center_y = (GRID_HEIGHT * HEXAGON_SIZE)
    shifted_mouse_pos = (mouse_pos[0] - SCREEN_WIDTH / 2 + offset_to_center_x, mouse_pos[1] - SCREEN_HEIGHT / 2 + offset_to_center_y)
    q, r = Hexagon.pixel_to_hex(shifted_mouse_pos, HEXAGON_SIZE)
    if 0 <= q < GRID_WIDTH and 0 <= r < GRID_HEIGHT:
        return hexagons[q][r]
    return None
    

def main():
    global selected_color
    grid = HexagonGrid(GRID_WIDTH, GRID_HEIGHT)
    hexagons = grid.grid

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    for i, button in enumerate(color_buttons):
                        if button.collidepoint(mouse_pos):
                            selected_color = COLOR_OPTIONS[i]
            elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                # Get the clicked hexagon and update its state
                mouse_pos = pygame.mouse.get_pos()
                clicked_hexagon = hexagon_clicked(mouse_pos, hexagons)
                update_hexagon_state(clicked_hexagon) 
        
        screen = update_screen(screen, hexagons)
        pygame.display.flip()
        clock.tick(90) # FPS
    
    pygame.quit()

if __name__ == "__main__":
    main()

