import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors

# Step 1: Create Adjacency Matrix (n x n)
n = 10
adjacency_matrix = np.zeros((n, n), dtype=int)

# Step 2: Visualize Hexagon Map
def hexagon(x, y, size):
    """Generate the vertices of a hexagon given center (x, y) and size."""
    angle = np.linspace(0, 2 * np.pi, 7)
    vertices = [(x + size * np.cos(a), y + size * np.sin(a)) for a in angle]
    return vertices

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axis('off')

hex_size = 1.0
colors = np.random.rand(n, n, 4)  # Color matrix with RGBA values

for i in range(n):
    for j in range(n):
        x = j * 1.5 * hex_size
        y = i * np.sqrt(3) * hex_size + j % 2 * np.sqrt(3) / 2 * hex_size
        hex_verts = hexagon(x, y, hex_size)
        hex_color = colors[i, j]
        hex_patch = RegularPolygon((x, y), numVertices=6, radius=hex_size, edgecolor='k', facecolor=hex_color)
        ax.add_patch(hex_patch)

plt.show()
