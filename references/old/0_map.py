import pygame as pg
import math
import old.AxialHexagon as hc

pg.init()

width, height = 800, 800
screen = pg.display.set_mode((width, height))
font = pg.font.Font(None, 36)  # You can replace 'None' with a font file path
pg.display.set_caption('Hexagonal Grid')

HEX_SIZE = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)

# def hex_to_pixel(q, r):
#     x = HEX_SIZE * 3/2 * q
#     y = HEX_SIZE * math.sqrt(3) * (r + q/2)
#     return int(x), int(y)

def cartesian_to_pixel(x, y):
    new_x = HEX_SIZE * 3/2 * x
    new_y = HEX_SIZE * math.sqrt(3) * (y + x/2)
    return int(new_x), int(new_y)

radius = 5
coordinates = hc.generate_hexagonal_coordinates(radius)

# Main loop
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill(WHITE)

    # Draw hexagons at mapped positions
    for coord in coordinates:
        x, y = hc.convert_to_cartesian(coord)
        old_x, old_y = cartesian_to_pixel(x, y)

        center_origin = lambda p: (p[0] + screen.get_width() // 2, p[1] + screen.get_height() // 2)
        x, y = center_origin((old_x, old_y))

        if old_x == 0 and old_y == 0:
            color = BLUE
        else:
            color = BLACK

        pg.draw.polygon(screen, color, [
            (x + HEX_SIZE, y),
            (x + HEX_SIZE / 2, y - HEX_SIZE * math.sqrt(3) / 2),
            (x - HEX_SIZE / 2, y - HEX_SIZE * math.sqrt(3) / 2),
            (x - HEX_SIZE, y),
            (x - HEX_SIZE / 2, y + HEX_SIZE * math.sqrt(3) / 2),
            (x + HEX_SIZE / 2, y + HEX_SIZE * math.sqrt(3) / 2)
        ], 2)

        # Write coordinates
        text = font.render(f"({q}, {r})", True, BLACK)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)

    pg.display.flip()

# Clean up
pg.quit()