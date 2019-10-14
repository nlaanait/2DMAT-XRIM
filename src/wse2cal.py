# This one is the second version of the calculation/simulation for WSe2/MoS2 multilayer scattering
#  Using real distance z (in Angstroms) instead of fraction of unit cell
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

write_file = 1
x_axis = "Q"
#x_axis = "L"
verbose= 1
plot_y = "lin" # "lin" or "log"

def cal_f0(coef, k):
  tmp = coef[5] 
  for ind in range(5):
    tmp = tmp + coef[ind]*np.exp(-coef[ind+6]*np.power(k, 2))
  return tmp

# For WSe2 layer
# Here are the atomic position along z axis, fractional 
#  based on half unit cell size, 3.2 Angstroms(?)
w_z1 = 0
se_z1 = -1.5576
se_z2 = 1.5576
# 2d = 6.49 Angstrom, layer spacing (W-W distance)
twod_wse2 = 6.49
# atomic position, in fraction of lattice
atompos_wse2 = np.array( [ [w_z1, se_z1, se_z2,] ] )
# calculate the atomic form factor for the specifc Q, using the Waaskirf coeffecients
coef_se = [17.354071,  4.653248,  4.259489,  4.136455,  6.749163, -3.160982, 2.349787, 0.002550, 15.579460, 45.181202,  0.177432,]
coef_w = [31.507900, 15.682498, 37.960129,  4.885509, 16.792112, -32.864574, 1.629485,  9.446448,  0.000898, 59.980675,  0.160798,]
# f' and f" are energy related, here use 10 keV.
fp_w = -10.9357
fp_se = -1.36
fpp_w= 3.9303    
fpp_se = 0.78
# vibrational amplitude, (in Angstroms)
vib_w  = .0
vib_se = .0

# For MoS2 layer
mo_z1 = 0
s_z1 = -1.5213
s_z2 = 1.5213
# layer spacing in MoS2
twod_mos2 = 5.945
# atomic position, in fraction of lattice
atompos_mos2 = np.array( [ [mo_z1, s_z1, s_z2,] ] )
# calculate the atomic form factor for the specifc Q, using the Waaskirf coeffecients
coef_s = [  6.372157,  5.154568,  1.473732,  1.635073,  1.209372,  0.154722,   1.514347, 22.092527,  0.061373, 55.445175,  0.646925,]
coef_mo = [  6.236218, 17.987711, 12.973127,  3.451426,  0.210899,  1.108770,   0.090780,  1.108310, 11.468720, 66.684151,  0.090780,]
# f' and f" are energy related, here use 10 keV.
fp_mo = -0.4820
fp_s = 0.2546
fpp_mo= 1.8715    
fpp_s = 0.3699
# vibrational amplitude, (in Angstroms)
vib_mo = .0
vib_s  = .0

# vibrational amplitude for each atom, in Angstroms, simply use 0.1 for now for all
atomvib_wse2 = np.array( [ [vib_w, vib_se, vib_se, ] ] )
atomvib_mos2 = np.array( [ [vib_mo, vib_s, vib_s, ] ] )
#atomvib_wse2 = np.array( [ [0., 0., 0., ] ] )
#atomvib_mos2 = np.array( [ [0., 0., 0., ] ] )

# number of single WSe2 layers in the structure, multiple calculations can be done
#nlayers_arr = [1, 2, 3]

# layer composition : 0- WSe2; 1 - MoS2.  Number of composition indicates number of layers.
# For example, [ [0], [0, 0], [0, 1, 1] ] is for 3 different calculations, 
#  the first one is one WSe2 layer
#  the second one is two WSe2 layers
#  the third one is two MoS2 layers on top of one WSe2 layer
#  
layer_comp = [ [0, ], [0, 1, ], [0, 1, 1] ]
# the corresponding layer positions for each set; the bottom layer is always 0;
#  and the next one is the offset from the bottom one in the unit of unit cell height
layer_pos = [ [0, 1.*twod_wse2, 2.*twod_wse2, ], [0, 1.*twod_wse2, 2.*twod_wse2,], 
  [0, 1.0*twod_wse2, 2.0*twod_wse2, 2.4*twod_wse2,] ]
# range of L, again based on the half unit cell size
Q = np.array(range(1, 521, 1))/100.0
Q_T = Q.reshape(-1, 1)
# translate to L
L = Q * twod_wse2 / (2*np.pi)

# here k = sin(theta)/LAMBDA = Q/(4*pi), for atomic form factor calculation
k = Q_T/(4*np.pi)
# atomic form factor calculation for WSe2
f0_se = cal_f0( coef_se, k )
f0_w = cal_f0( coef_w, k )
fp_wse2 = np.tile( [fp_w, fp_se, fp_se,], Q_T.shape )
fpp_wse2 = np.tile( [fpp_w, fpp_se, fpp_se,], Q_T.shape )
atomFormFactor_wse2 = np.column_stack( (f0_w, f0_se, f0_se) ) + fp_wse2 + 1j*fpp_wse2
#print "size of fp_wse2 is %d x %d: " %(fp_wse2.shape)
#print "size of fpp_wse2 is %d x %d: " %(fpp_wse2.shape)
# atomic form factor calculation for WSe2
f0_s = cal_f0( coef_s, k )
f0_mo = cal_f0( coef_mo, k )
fp_mos2 = np.tile( [fp_mo, fp_s, fp_s,], Q_T.shape )
fpp_mos2 = np.tile( [fpp_mo, fpp_s, fpp_s,], Q_T.shape )
atomFormFactor_mos2 = np.column_stack( (f0_mo, f0_s, f0_s) ) + fp_mos2 + 1j*fpp_mos2

fig = plt.figure(1)

_myline =[]
vline_x=[]
color_list = ["b", "r", "g", "c", "m", "k"]
for (layerc, layerp, mycolor) in zip(layer_comp, layer_pos, color_list):
  atomp = np.array([])
  atomff = np.array([])
  atomv = np.array([])
  # build all the atomic positions for n_layer of WSe2
  curvelabel = ""
  for layer in range(len(layerc)):
    # this is the flag indicating WSe2/MoS2
    flag = layerc[layer]
    # build atomic positions for this layer and add to the whole structure 
    atomp_thislayer = atompos_mos2 if flag else atompos_wse2
    atomp_thislayer = atomp_thislayer + layerp[layer]
    atomp = np.hstack([atomp, atomp_thislayer]) if atomp.size else atomp_thislayer
    
    # build atomic form factor for layers 
    atomff_thislayer = atomFormFactor_mos2 if flag else atomFormFactor_wse2
    atomff = np.hstack([atomff, atomff_thislayer]) if atomff.size else atomff_thislayer
    
    # build atom vibration amplitude
    atomv_thislayer = atomvib_mos2 if flag else atomvib_wse2
    atomv = np.hstack([atomv, atomv_thislayer]) if atomv.size else atomv_thislayer

    # setup the plot label
    thislabel = "$MoS_2$" if flag else "$WSe_2$"
    curvelabel = curvelabel + "_" + thislabel if len(curvelabel)!=0 else thislabel

  if verbose:
    print "atomic positions:"
    print(atomp)

  # exp(-i*Q.r)
  tmp = np.matmul(Q_T, atomp)
  tmp = np.exp(-1j*tmp)
  # exp(-Q^2*u^2) -- Debye-waller factor
  tmp1 = np.matmul(Q_T, atomv)
  tmp1 = np.exp(-np.power(tmp1, 2)/2)

  #atomff = np.tile(atomff, Q_T.shape)
  f = np.multiply(np.multiply(atomff, tmp), tmp1)

  F = np.sum(f, axis = 1)
  I = np.square( np.absolute(F) )
  vline_x = vline_x + [ Q[np.argmax(I)] ]
  if verbose:
    print "maximum position for intensities:"
    print(vline_x)
  

  if write_file:
    if x_axis == "Q":
      x_data = Q
    elif x_axis == "L":
      x_data = L
    else:
      print "X axis has to be either 'Q' or 'L'. "
      exit()
    data = np.transpose( np.append([x_data], [I, np.sqrt(I)], axis = 0) )
    #print data.shape
    output_filename = "wse2cal_%d.xye" % (layer)
    with open(output_filename, 'w') as outfile:
      outfile.write('# Layer number in WSe2 = %d\n' % layer)
      outfile.write('# %s Intensity Error\n' % x_axis)
      np.savetxt(outfile, data, fmt='%10.6e %10.6e %10.6e')
      if verbose:
        print "Data written to " + output_filename + "."
      outfile.close()

  # Now the plotting part.
  _thishandle, = plt.plot(x_data, I, marker = '.', color = mycolor, label = curvelabel)
  _myline.extend([_thishandle])
  h_a = plt.gca()
  if plot_y is 'lin':
    h_a.set_yscale('linear')
  elif plot_y is 'log':
    h_a.set_yscale('log')
  else:
    print "Invalid y-axis scaling, must be 'lin' or 'log'."
    sys.exit()
  
plt.title( "Multiple $WSe_2$/$MoS_2$ layers scattering intensity" , fontsize = 16)
if x_axis == "Q":
  plt.xlabel("Q ($\AA^{-1}$)", fontsize=14, color="red")
elif x_axis =="L":
  plt.xlabel("L (rlu)", fontsize=14, color="red")

#vline_x = [0.75, 0.80]
for (vline_x0, mycolor) in zip(vline_x, color_list):
  plt.axvline(x=vline_x0,  linestyle='dashed', color = mycolor)
plt.ylabel('$|F|^2$ [arb. units]', fontsize=14, color="red")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(handles=_myline)
plt.show()
  
