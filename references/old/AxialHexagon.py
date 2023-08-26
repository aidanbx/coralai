import math

class AxialHexagon(object):
    
    def __init__(self, q, r):
        self.q = q
        self.r = r
        self.neighbors = []
        self.color = ""

    def __repr__(self):
        return f'(q={self.q}, r={self.r})'
    
    def get_neighbors(self):
        hex_directions = [(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)]
        neighbors = []
        for direction in hex_directions:
            neighbors.append((self.q + direction[0], self.r + direction[1]))


class AxialHexagonGrid(object):

    def __init__(self, radius):
        self.radius = radius
        self.hexagons = self.generate_hexagons(radius)

    def generate_hexagons(radius):
        hexagons = []
        hexagons.append(AxialHexagon(0, 0))
        
        for r in range(0, -radius, -1):
            for q in range(-r - 1, -radius - r, -1):
                hexagon = AxialHexagon(q, r)
                hexagons.append(hexagon)

        for r in range(1, radius):
            for q in range(0, -radius, -1):
                hexagon = AxialHexagon(q, r)
                hexagons.append(hexagon)
                
        for q in range(1, radius):
            for r in range(-q, radius - q):
                hexagon = AxialHexagon(q, r)
                hexagons.append(hexagon)

        return hexagons

