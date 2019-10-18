import xrayutilities as xu
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.cluster import KMeans, MiniBatchKMeans

def build_crystal(layer_seq=OrderedDict(), substrate=None, mos2_layer=None, wse2_layer=None, 
                  vacuum=None, stack_name='', substrate_spacing=1.25, roughness=0.0, mos2_thickness=6, wse2_thickness=6.5):
    '''
    Build crystal specified by layer sequence ordered dictionary.
    e.g. from top to bottm 1 x mos2 + 1 x wse2 + 3 x wse2
    layer_seq = OrderedDict({"layer_1": {"mos2": 1}, "layer_2": {"wse2": 1}, "layer_3": {"wse2": 3}})
    '''
    substrate.roughness = roughness
    wse2_layer.roughness = 0.0 # this is adhoc- but could be easily fitted as well
    wse2_layer.thickness = wse2_thickness
    mos2_layer.roughness = 0.0
    mos2_layer.thickness = mos2_thickness
    vacuum = xu.materials.Amorphous('H', 1e-10)
    layers = [substrate]
    layers += [xu.simpack.Layer(vacuum, substrate_spacing, roughness=0.0)]
    for name, specs in layer_seq.items():
        if list(specs.keys())[0] == "mos2":
            for _ in range(specs["mos2"]):
                layers += [mos2_layer] 
        elif list(specs.keys())[0] == "wse2":
            for _ in range(specs["wse2"]):
                layers += [wse2_layer]
        else:
            t = specs["vacuum"]
            vacuum_layer = xu.simpack.Layer(vacuum, t, roughness=0.0)
            layers += [vacuum_layer]
            
    args = ["layers[%d]+"%d for d in range(len(layers))]
    args = ''.join(args)
    args = args[::-1].replace('+','',1)
    args = args[::-1]
    return xu.simpack.LayerStack("stack_%s" % str(stack_name), eval(args))       

def simFunc(x, *args, **kwargs):
    # add a new parameter
    spacing_1, spacing_2, spacing_3, roughness, layer_occ_1, layer_occ_2, scale, q_offset, wse2_t, mos2_t = x
    layer_occ_1 = int(layer_occ_1)
    layer_occ_2 = int(layer_occ_2)
    layer_occ_2 = layer_occ_2 if layer_occ_1 != 0 else 0
    xrr_data, q_z, crystal_params = args[:]
    layer_dict = OrderedDict({"layer_1": {"wse2": 1}, 
                          "layer_2": {"vacuum": spacing_2}, 
                          "layer_3": {"mos2":layer_occ_1},
                          "layer_4": {"vacuum": spacing_3}, 
                          "layer_5": {"mos2": layer_occ_2}})
    # build film
    #pass new params
    film = build_crystal(layer_seq=layer_dict, substrate=crystal_params[0], mos2_layer=crystal_params[1], 
                         wse2_layer=crystal_params[2], vacuum=crystal_params[3], stack_name="1_1_1", 
                         substrate_spacing=spacing_1, roughness=roughness, mos2_thickness=mos2_t, wse2_thickness=wse2_t)
    # simulate
    en = 10000
    resolution = np.rad2deg(2e-3)
    m = xu.simpack.SpecularReflectivityModel(film, sample_width=10, beam_width=0.3, energy=en, I0=scale)
#                                              resolution_width=resolution, I0=1e4)
    q_z_shifted = np.copy(q_z) + q_offset
#     print("q_z_shift:", q_z_shifted.shape)
    alpha = np.rad2deg(np.arcsin(xu.en2lam(en) * q_z_shifted / (4* np.pi))).squeeze()
#     print("alpha:", alpha.shape)
    xrr_sim = m.simulate(alpha)
    del film, layer_dict
    if 'fit_return' in list(kwargs.keys()):
        return xrr_sim, m.densityprofile(300, plot=False)
    return xrr_sim


def objFunc(x, *args, **kwargs):
    xrr_sim = simFunc(x, *args, **kwargs)
    xrr_data = args[0]
    if 'fit_return' in list(kwargs.keys()):
        return xrr_sim
    else: 
        fom = np.sum(np.abs(xrr_sim - xrr_data))
        if np.isnan(fom): fom = 100.
        return fom
    
def cluster_lscans(data_arr, num_clusters=8):
    data = np.copy(data_arr)
    data = data.transpose(1,2,0)
    orig_shape = data.shape
    data = data.reshape(-1, data.shape[-1])
    # cluster data
    # kmeans_data = KMeans(n_clusters=num_clusters, n_jobs=-1, random_state=0)
    kmeans_data = MiniBatchKMeans(n_clusters=num_clusters, random_state=0) 
    kmeans_data.fit_predict(data)
    # count # of traces per cluster
    print('Number of L-scans per cluster: \n')
    print([np.where(kmeans_data.labels_ == i)[0].size for i in range(num_clusters)])
    
    labels_imp = kmeans_data.labels_.reshape(orig_shape[:-1])
    
    return labels_imp, kmeans_data.cluster_centers_, kmeans_data.labels_
    
    
    
    
