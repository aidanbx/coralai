# Axial Coordinates
import math

BLACK = (0, 0, 0)
DARK_GREY = (30, 30, 30)
OFF_WHITE = (255, 255, 240)
DARK_BEIGE = (139, 121, 94)
DARK_MAGENTA = (139, 0, 139)
MAROON = (128, 0, 0)
FOREST_GREEN = (34, 139, 34)
YELLOW = (255, 255, 0)
NAVY_BLUE = (0, 0, 128)


class Hexagon:
    def __init__(self, q, r, color=OFF_WHITE):
        self.q = q
        self.r = r
        self.x = None
        self.y = None
        self.color = color
        self.neighbors = []

    def __repr__(self):
        return f'({self.q}, {self.r})'

    def add_neighbor(self, key):
        self.neighbors.append(key)
    
    def get_vertices(self, hexagon_size):
        self.x = hexagon_size * 3/2 * self.q
        self.y = hexagon_size * (math.sqrt(3) / 2 * self.q + math.sqrt(3) * self.r)
        
        vertices = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            vertex_x = self.x + hexagon_size * math.cos(angle_rad)
            vertex_y = self.y + hexagon_size * math.sin(angle_rad)
            vertices.append((vertex_x, vertex_y))
        
        return vertices

    

class HexagonGrid:
    def __init__(self, radius, hexagon_size):
        self.radius = radius
        self.hexagon_size = hexagon_size
        self.hexagons = self.generate_hexagons() # dict
        
    def generate_hexagons(self):
        radius = self.radius
        hexagons = {}
        hexagons["0,0"] = Hexagon(0, 0)
        
        for r in range(0, -radius, -1):
            for q in range(-r - 1, -radius - r, -1):
                hexagons[f"{q},{r}"] = Hexagon(q, r)

        for r in range(1, radius):
            for q in range(0, -radius, -1):
                hexagons[f"{q},{r}"] = Hexagon(q, r)
                
        for q in range(1, radius):
            for r in range(-q, radius - q):
                hexagons[f"{q},{r}"] = Hexagon(q, r)

        hexagons = self.generate_adjacency_matrix(hexagons)

        return hexagons
    
    def generate_adjacency_matrix(self, hexagons):
        directions = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]

        for key in hexagons: 
            hexagon = hexagons[key]
            for direction in directions:
                new_key = f"{hexagon.q+direction[0]},{hexagon.r+direction[1]}"
                if new_key in hexagons:
                    hexagons[key].add_neighbor(new_key)

        return hexagons

    # Specificially for pygame coords
    def hexagon_clicked(self, mouse_pos, screen_width, screen_height): 
        # Convert mouse position to relative position within the grid
        relative_x = mouse_pos[0] - screen_width / 2
        relative_y = mouse_pos[1] - screen_height / 2

        # Calculate axial coordinates based on relative position and hexagon size
        q = (2 / 3 * relative_x) / self.hexagon_size
        r = (-1 / 3 * relative_x + math.sqrt(3) / 3 * relative_y) / self.hexagon_size

        # Iterate through hexagons and find the clicked hexagon
        for hexagon in self.hexagons.values():
            if hexagon.q == round(q) and hexagon.r == round(r):
                return hexagon

        return None




# Example usage
# grid = HexagonGrid(radius=3)
# for key in grid.hexagons:
#     hex = grid.hexagons[key]
#     neighbors = hex.neighbors
#     print(f"{key}: {neighbors}")









