import numpy as onp
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.experimental.host_callback import id_print
from jax.flatten_util import ravel_pytree

import matplotlib.pyplot as plt
from functools import partial

import os
import subprocess

import jax

# Making predictions on CPU (Select a GPU device for GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
print("JAX Cuda Devices: ", jax.local_devices())

# Set JAX to use double precision # Comment this out if not
#jax.config.update('jax_enable_x64', True)


# Local imports
from deeponet_models import DeepONetModel, ANNModel




"""
**Directory where these files need to placed**

- Ymin.npy, Ymax.npy (Min Max files for selected scalars)
- Pre/Post DeepONet Trained Parameters
- TempNet Trained Parameters
- iEneMin, iEnemax  (Min Max for internal energy)
- Correlation Net Trained Parameters
- outMin, outMax (Min Max files for left out species)
- YiEneMin, YiEneMax (Min Max files for selected scalars, temperature replaced with internal energy)

"""

#data_path = f'/mnt/beegfs/akumar35/DeepONet_Data/URANS_Clean/Local_Prediction/'
data_path = f'Local_Prediction/'

Ymin = onp.load(data_path+"Ymin.npy")
Ymax = onp.load(data_path+"Ymax.npy")

tempMin = Ymin[0]
tempMax = Ymax[0]

iEneMin = onp.load(data_path+"iEneMin.npy")
iEneMax = onp.load(data_path+"iEneMax.npy")

outMin = onp.load(data_path+"outMin.npy")
outMax = onp.load(data_path+"outMax.npy")

YiEneMin = Ymin.copy()
YiEneMin[0] = iEneMin
YiEneMax = Ymax.copy()
YiEneMax[0] = iEneMax



# Defining parameters
nselect = 10 # number of scalars for evolution
npower = 5  # scaling power (1/npower)
loc_dt = 5E-4  # dt in miliseconds , make sure this same as in the training code
ptlarge = 4 # make sure it is same as in the training code

t_max = loc_dt*(ptlarge-1) # maximum dt possible same as window size in milliseconds, used in scaling dt
t_min = 0.0 # minimum dt possible, used in scaling dt

nselectout = len(outMin) # Length of left-out vector to be predicted
print(nselectout)



""" Make sure model parameters defined here are same as training"""
# Instantiation of DeepONet model
rng_key=random.PRNGKey(1234)
pre_key, ref_key, rates_key, mod_key = random.split(rng_key, 4)

nlayer = 60
branch_layers = [nselect, 150, 150, 150, 150, nselect*nlayer]
trunk_layers =  [1,       150, 150, 150, 150, nselect*nlayer]
#param_layers = [1, 150, nselect*nlayer]  # Uncomment it if you want Param Net

param_layers_defined = locals().get('param_layers', None)

if param_layers_defined:
    pre_weights_name = f'param_pre_deeponet_weights.npy'
    refined_weights_name = f'param_refined_deeponet_weights.npy'
    rates_weights_name = f'param_rates_deeponet_weights.npy'
else:
    pre_weights_name = f'pre_deeponet_weights.npy'
    refined_weights_name = f'refined_deeponet_weights.npy'
    rates_weights_name = f'rates_deeponet_weights.npy'

optimizer = optimizers.adamax(1e-3)
activation = np.tanh
model = DeepONetModel(branch_layers, trunk_layers, optimizer, activation, mod_key, param_layers=param_layers_defined)

print(f'Total Number of Model Parameter: {len(ravel_pytree(model.get_params(model.opt_state))[0])}\n')
    
# Loading refined training parameters
autoreg_step = 60# [10,20,40,60,90,120]
filename = f"{data_path}PreDNet/{autoreg_step}_{refined_weights_name}"
#filename = f"{data_path}Pre32DNet/{autoreg_step}_{refined_weights_name}"
#filename = f"/mnt/beegfs/akumar35/DeepONet_Data/URANS_Clean/PrePost/Pre/Model_Data/{autoreg_step}_{refined_weights_name}"
print(f'PreDeepONet Param File being loaded as: {filename}\n')
ref_flat_params = np.load(filename)
_, unravel = ravel_pytree(model.get_params(model.opt_state))
pre_refined_params = unravel(ref_flat_params)

autoreg_step = 60# [10,20,40,60,90,120]
#filename = f"{data_path}PostDNet/{autoreg_step}_{refined_weights_name}"
filename = f"{data_path}Post32DNet/{autoreg_step}_{refined_weights_name}"
#filename = f"/mnt/beegfs/akumar35/DeepONet_Data/URANS_Clean/PrePost/Post/Model_Data/{autoreg_step}_{refined_weights_name}"
print(f'PostDeepONet Param File being loaded as: {filename}\n')
ref_flat_params = np.load(filename)
_, unravel = ravel_pytree(model.get_params(model.opt_state))
post_refined_params = unravel(ref_flat_params)


# Instantiation of TempNet and Rec Net
rng_key = random.PRNGKey(1111)
key1, key2, key3, key4 = random.split(rng_key, 4)

activation = np.tanh
temp_layers = [nselect, 10, 10, 1]
tempnet_model = ANNModel(temp_layers, optimizer, activation, key3)
print(f'Total Number of TempNet Model Parameter: {len(ravel_pytree(tempnet_model.get_params(tempnet_model.opt_state))[0])}\n')

filename = f"{data_path}tempnet_params.npy"
print(f'File being loaded as: {filename}\n')
flat_params = np.load(filename)
_, unravel = ravel_pytree(tempnet_model.get_params(tempnet_model.opt_state))
tempnet_params = unravel(flat_params)

#RecNet
activation = np.tanh
rec_layers = [nselect, 30, 30, nselectout]
recnet_model = ANNModel(rec_layers, optimizer, activation, key4)
print(f'Total Number of RecNet Model Parameter: {len(ravel_pytree(recnet_model.get_params(recnet_model.opt_state))[0])}\n')

filename = f"{data_path}recnet_params.npy"
print(f'File being loaded as: {filename}\n')
flat_params = np.load(filename)
_, unravel = ravel_pytree(recnet_model.get_params(recnet_model.opt_state))
recnet_params = unravel(flat_params)   







""" DeNormalizing output vector of leftout scalars. """
@jit
def descale_leftout(Yin):
    Yin = (Yin + 1.0)/2.0
    Yout = Yin*(outMax - outMin) + outMin
    Yout = np.power(Yout, npower)
    return Yout 
    
""" DeNormalizing output vector of selected scalars. """
@jit
def descale_select(Yin):
    Yin = (Yin + 1.0)/2.0
    Yout = Yin*(Ymax - Ymin) + Ymin  
    Yout = np.power(Yout, npower)
    return Yout 




# Define temperature for separating Pre/Post DeepONets predictions
temp_shift = 1010 #1110 #1010
val_shift = 2.0*(np.power(temp_shift, 1/npower) - tempMin)/(tempMax-tempMin)-1.0

print(f'\nDividing Temperature: {temp_shift}K with Normal value: {val_shift}\n')





""" Normalizing input vector. """
@jit
def inp_scale(Yin, dt):
    Yin = np.power(Yin, 1/npower)
    X = 2.0 * (Yin - Ymin) / (Ymax - Ymin) - 1.0
    dtn = 0.5 * (dt - t_min) / (t_max - t_min)
    return X, dtn

""" Normalizing input vector and finding local temperature. """
@jit
def inp_scale_ien(Yin, dt):
    Yin = np.power(Yin, 1/npower)
    X = 2.0 * (Yin - YiEneMin) / (YiEneMax - YiEneMin) - 1.0

    temp_val = tempnet_model.predict_ann(tempnet_params, X) # local temperature
    X = X.at[0].set(temp_val[0])
    dtn = 0.5 * (dt - t_min) / (t_max - t_min)
    
    return X, dtn



""" Combined Model Predictions Without Internal Energy """
@jit
def reactdeeponet_predict(u0, dt, equiv):

    dt = dt * 1E3 # Converting to ms
    
    u0n, dtn = inp_scale(u0, dt)
    
    temp = u0n[0]
    
    def true_fun(_):
        return model.predict_deeponet(pre_refined_params, u0n, dtn, equiv)
    
    def false_fun(_):
        return model.predict_deeponet(post_refined_params, u0n, dtn, equiv)

    # If else condition in JAX
    out_pred = jax.lax.cond(temp < val_shift, None, true_fun, None, false_fun)

    # Reconstructing from Correlation Net
    leftout_pred = recnet_model.predict_ann(recnet_params, out_pred)

    # Denormalizing leftout scalars
    leftout_mf = descale_leftout(leftout_pred)
    # Denormalizing selected scalars
    select_mf = descale_select(out_pred)  
    
    all_mf = np.hstack((select_mf, leftout_mf)) 
    #all_mf = select_mf
    
    return all_mf


    
""" Combined Model Predictions With Internal Energy """
@jit
def ien_reactdeeponet_predict(u0, dt, equiv, sie):

    dt = dt * 1E3 # Converting to ms
    sie = sie + 2E6 # Constant value added as done in training part (removing negative values)
    u0 = np.append(sie, u0)
   
    u0n, dtn = inp_scale_ien(u0, dt)
    
    temp = u0n[0]
    
    def true_fun(_):
        return model.predict_deeponet(pre_refined_params, u0n, dtn, equiv)
    
    def false_fun(_):
        return model.predict_deeponet(post_refined_params, u0n, dtn, equiv)

    # If else condition in JAX
    out_pred = jax.lax.cond(temp < val_shift, None, true_fun, None, false_fun)

    # Reconstructing from Correlation Net
    leftout_pred = recnet_model.predict_ann(recnet_params, out_pred)

    # Denormalizing leftout scalars
    leftout_mf = descale_leftout(leftout_pred)
    # Denormalizing selected scalars
    select_mf = descale_select(out_pred)  
    
    all_mf = np.hstack((select_mf, leftout_mf)) 
    
    return all_mf


# Just for Converting to Numpy Arrays to communicate with Converge
def predict_from_c(u0, dt, equiv, sie):

    a = ien_reactdeeponet_predict(u0, dt, equiv, sie)
    #a = reactdeeponet_predict(u0, dt, equiv, sie)
    a_out = onp.array(a)
    return a_out

def predict_from_c_vmap(u0, dt, equiv, sie):

    a = vmap(ien_reactdeeponet_predict, (0,None, None,0))(u0,dt,equiv,sie)
    #a = vmap(reactdeeponet_predict, (0,None, None,0))(u0,dt,equiv,sie)
    a_out = onp.array(a)
    return a_out


# Local check
dt = 1E-6
sie = 1246205.3039284516 - 2E6
equiv = 1.0 # Not utilized here
u0 = onp.array([1.23015463e-01, 3.59179936e-02, 1.45457045e-03, 5.54114394e-02, 2.48897239e-04, 5.17131877e-04, 3.58152378e-04, 8.22510570e-02, 2.72807060e-03])
out_mf = predict_from_c(u0, dt, equiv, sie)
print(f'Local Check: \nLength of output: {len(out_mf)} \nOutput values: {out_mf[0:15]}\n')

