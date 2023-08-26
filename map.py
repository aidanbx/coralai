import pygame
from Hexagon import HexagonGrid
from Hexagon import BLACK, DARK_GREY, OFF_WHITE, DARK_BEIGE, DARK_MAGENTA, MAROON, FOREST_GREEN, YELLOW, NAVY_BLUE

COLOR_OPTIONS = [OFF_WHITE, BLACK, DARK_BEIGE, DARK_MAGENTA, MAROON, FOREST_GREEN, NAVY_BLUE]

RADIUS = 20
HEXAGON_SIZE = 10
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 800
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

def update_screen(screen, hexagons):
    screen.fill(DARK_GREY)  # Clear the screen with dark_grey color
    
    for i, button in enumerate(color_buttons): 
        if COLOR_OPTIONS[i] == selected_color:
            pygame.draw.rect(screen, selected_color, button)
            pygame.draw.rect(screen, YELLOW, button, 3)
        else:
            pygame.draw.rect(screen, COLOR_OPTIONS[i], button)
    
    # Draw the hexagons
    for hexagon in hexagons.values():
        vertices = hexagon.get_vertices(HEXAGON_SIZE)
        shifted_vertices = [(x + screen.get_width() / 2, y + screen.get_height() / 2) for x, y in vertices]
        pygame.draw.polygon(screen, hexagon.color, shifted_vertices, 0)
        pygame.draw.polygon(screen, BLACK, shifted_vertices, 1)
        
    return screen

def main():
    global selected_color
    grid = HexagonGrid(RADIUS, HEXAGON_SIZE)
    hexagons = grid.hexagons

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
                clicked_hexagon = grid.hexagon_clicked(mouse_pos, SCREEN_WIDTH, SCREEN_HEIGHT)
                update_hexagon_state(clicked_hexagon) 
        
        screen = update_screen(screen, hexagons)
        pygame.display.flip()
        clock.tick(90) # FPS
    
    pygame.quit()

if __name__ == "__main__":
    main()

