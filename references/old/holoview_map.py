import holoviews as hv
from holoviews import opts

import old.AxialHexagon as hexagon

hv.extension('bokeh')


def hex_tile_app(doc):
    hexagon_data = hexagon.generate_hexagons_in_radius(3)

    hex_tile_map = hv.HexTiles(hexagon_data)

    hex_tile_map.opts(
        cmap='Category10',
        width=800,
        height=800,
        tools=['hover'],
    )

    hv.renderer('bokeh').server_doc(hex_tile_map)


if __name__ == '__main__':
    from bokeh.server.server import Server

    server = Server({'/': hex_tile_app}, num_procs=1)
    server.start()

    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print("Server stopped")
