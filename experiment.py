import numpy as np
import taichi as ti
import torch
from eincasm.Eincasm import Experiment
from eincasm.vis.PixelVis import PixelVis
from eincasm.vis.Vis import Vis

ti.init(ti.gpu)      
ein = Eincasm(shape=(100,100), torch_device=torch.device('mps'))

w = ein.world

vis = Vis(ein, [('com', 'r'), ('com', 'g'), ('com', 'b')])

while vis.window.running:
    v = vis.vars[None]
    if not v.drawing:
        ein.apply_physics()
        ein.organism.apply_weights()
    vis.update()


























exit()
ein = eincasm(shape=(50,50), torch_device=torch.device('mps'))

ein.world['capital'] = torch.arange(torch.numel(ein.world['capital'])).reshape(ein.world['capital'].shape)
ein.world['waste'] = (torch.arange(torch.numel(ein.world['waste']))*10).reshape(ein.world['waste'].shape)

@ti.func
def process_cell(mr: ti.f32, mg: ti.f32):
    return mr*mg

@ti.kernel
def test(mem: ti.types.ndarray(), inds: ti.template()):
    for i,j in ti.ndrange(mem.shape[0], mem.shape[1]):
        capital = mem[i,j, inds.capital]
        for mid in ti.static(range(inds.muscles.shape[0])):
            mem[i,j, inds.port] = process_cell(mem[i,j, inds.capital], mem[i,j, inds.waste])

test(ein.world.mem, ein.world.ti_indices)
exit()

mem = [[1,2,3,4,5,6,7,8,9],
       [1,2,3,4,5,6,7,8,9],
       [1,2,3,4,5,6,7,8,9]]

mem = np.array(mem, dtype=np.float32)
mem[1] *= 10
mem[2] *= 100

ti_inds = {}
ti_inds_types = {}
mids = [0,1]
pids = [3,5]
cid = [8]

def add_ti_inds(inds, chid):
    inds=np.array(inds)
    inds_vec = ti.Vector(inds)
    inds_vec_type = ti.types.vector(n=inds.shape[0], dtype=ti.i32)
    ti_inds_types[chid] = inds_vec_type
    ti_inds[chid] = inds_vec

add_ti_inds(mids, 'mids')
add_ti_inds(pids, 'pids')
add_ti_inds(cid, 'cid')

ti_inds_struct_type = ti.types.struct(**ti_inds_types)
ti_inds_field = ti_inds_struct_type.field(shape=())
for chid in ti_inds:
    ti_inds_field[None][chid] = ti_inds[chid]

@ti.func
def tfunc(m: ti.f32, p: ti.f32, c: ti.f32):
    return m+p+c

@ti.kernel
def test(mem: ti.types.ndarray(), ti_inds: ti_inds_struct_type):
    for i in ti.ndrange(mem.shape[0]):
        cid = ti_inds.cid
        c = mem[i, cid]
        # for idx in ti.ndrange(ti_inds.mids.shape[0]):
        #     m = mem[i, ti_inds.mids[idx]]
        #     p = mem[i, ti_inds.pids[idx]]
        #     mem[i, ti_inds.mids[idx]] = tfunc(m, p, c)

test(mem, ti_inds_field[None])
print(mem)
exit()

ein = eincasm(shape=(50,50), torch_device=torch.device('mps'))
ein.world['capital'] = torch.ones_like(ein.world['capital'])
ein.world.windex_obj.return_ti(True)
ids = ein.world.windex_obj

@ti.func
def tfunc(capital: ti.f32, muscle: ti.f32, growth_act: ti.f32):
    return capital+muscle+growth_act, capital*muscle*growth_act

# @ti.func
# def tfunc(mem: ti.types.ndarray(), cid: ti.template(), mids: ti.template()):
#     for i,j in ti.ndrange(mem.shape[0], mem.shape[1]):
#         for mid in range(mids.shape[0]):
#             mem[i, j, mids[mid]] += mem[i, j, cid[0]]

@ti.func
def route_ids(ids: ti.template()):
    return ids

@ti.kernel
def test(mem: ti.types.ndarray()):
    # mids = route_ids(ids['muscles'])
    # cid = route_ids(ids['capital'])
    # gid = route_ids(ids['growth_act'])
    for i,j in ti.ndrange(mem.shape[0], mem.shape[1]):
        for idx in ti.ndrange(ids['muscles'].shape[0]):
            capital = mem[i, j, cid]
            muscle = mem[i, j, mids[idx]]
            growth_act = mem[i, j, gid[idx]]
            dcap, dmus = tfunc(capital, muscle, growth_act)
            mem[i, j, cid] += dcap
            mem[i, j, mids[idx]] += dmus

@ti.kernel
def test(mem: ti.types.ndarray()):
    tfunc(mem, ein.world.windex_obj['capital'], ein.world.windex_obj['muscles'])

test(ein.world.mem)
print(ein.world['muscles'])

exit()
ein = eincasm(shape=(50,50), torch_device=torch.device('mps'))

def update_world():
    ein.apply_ti_physics(ein.world.mem)

vis = PixelVis(ein.world, {
    'capital': 'viridis',
    'waste': 'hot',
    'port': 'copper',
    'obstacle': 'gray'
}, update_world)

print(ein.world['capital'].shape)
print(vis.chlims)
vis.launch()
print(ein.world['obstacle'].min(), ein.world['obstacle'].max())