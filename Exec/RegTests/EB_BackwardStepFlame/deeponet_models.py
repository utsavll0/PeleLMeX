import jax.numpy as np
from jax import random, grad, vmap, jit, lax, jacrev
from jax.nn import softmax
from jax.scipy.special import logsumexp
from functools import partial
import itertools
from tqdm import trange


from NN_architectures import gatingMLP, DeepONet, modifiedMLP





class DeepONetModel:
    """ DeepONet model with loss formulations and training functionalities """

    def __init__(self, branch_layers, trunk_layers, optimizer, activation=np.tanh, rng_key=random.PRNGKey(0), param_layers=None):
        """Initializes DeepONet model"""
        
        self.rng_key = rng_key

        # Instantiate of DeepONet with a unique RNG key
        self.model_DeepONet = DeepONet(branch_layers, trunk_layers, activation, rng_key, param_layers)

        # Initialize optimizer with parameters
        aggregated_params = self.model_DeepONet.params
        
        self.opt_init, self.opt_update, self.get_params = optimizer
        self.opt_state = self.opt_init(aggregated_params)

        # Logging utilities
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_log_ref = []
        self.loss_rates_log = []
        self.loss_onlyrates_log = []
        self.loss_valuerates_log = []

        
    
    # DeepONet Forward pass       
    def forward_pass(self, params, u, y, e):
        """Computes forward pass through DeepONet """
        
        # Individual predictions from each DeepONet
        G = self.model_DeepONet.predict(params, u, y, e)

        return G


    # Comparing point wise predictions for pre training
    def value_loss(self, params, batch, training_param_tuple):
        """
        Computes the loss for a batch of data based on model predictions and ground truth values.
        """

        (loss_p, w_rates) = training_param_tuple
        
        inputs, outputs = batch
        
        #Unpack `inputs` while handling the optional `e` parameter
        if len(inputs) == 3:
            u, y, e = inputs
        else:
            u, y = inputs
            e = None

        out_pred = vmap(self.forward_pass, (None, 0, 0, 0))(params, u, y, e)

        loss = np.mean(np.abs((outputs - out_pred))**loss_p)
        
        return loss

    

    # Output Jacobian wrt to input of Trunk Net
    def out_grad(self, params, u, y, e):

        """
        Evaluates output Jacobian wrt to input to Trunk Net
        
        """
        
        def temp_out(params, u, y, e):
            g_out= self.forward_pass(params, u, y, e)
            t_out= g_out[0]
            
            return t_out

        # Gives full Jacobian matrix
        out_jac = jacrev(lambda x: self.forward_pass(params, u, x, e))(y)

        # Only Temp Grad can be obtained like this as well
        dTdt=  grad(temp_out, argnums=2)(params, u, y, e)

        # Another way to obtain Temp Grad
        #dTdt= out_jac[0]
        
        return dTdt

    
    def only_rates_loss(self, params, batch, training_param_tuple):
        
        """
        Finds reaction rates loss. 
        Only Temp is considered for now.
        
        """
        (loss_p, w_rates) = training_param_tuple
        
        inputs, outputs= batch

        # Only Temp Grad is considered
        outputs= outputs[:,0]
        
        #Unpack `inputs` while handling the optional `e` parameter
        if len(inputs) == 3:
            u, y, e = inputs
        else:
            u, y = inputs
            e = None
        
        out_pred = vmap(self.out_grad, (None, 0, 0, 0))(params, u, y, e)
        
        r_loss = np.mean(np.abs((outputs - out_pred))**loss_p)
        
        return r_loss

    
    # Definition of Rates Loss
    def rates_loss(self, params, value_batch, rates_batch, training_param_tuple):
        """
        Computes the total loss.
        Value_loss + Rates_loss
        
        """
        (loss_p, w_rates) = training_param_tuple
        # Compute value loss
        tot_loss = self.value_loss(params, value_batch, training_param_tuple)
        
        #Compute rates loss
        r_loss = self.only_rates_loss(params, rates_batch, training_param_tuple)
        
        tot_loss = tot_loss + w_rates*r_loss
        
        return tot_loss


    # Refined training with autoregressive rollout
    def refine_loss(self, params, batch, training_param_tuple, autoreg_steps):
        """
        Computes the refinement loss over a given number of autoregressive steps.
        
        Parameters:
        - params: The parameters of the MOE DeepONet model.
        - batch: A batch of data including inputs and expected outputs.
        - autoreg_steps: The number of autoregressive steps to predict into the future.
        
        Returns:
        - The computed refinement loss.
        """
        (loss_p, w_rates) = training_param_tuple
        
        inputs, outputs = batch
        # Transpose outputs for batch processing
        outputs = np.transpose(outputs, (1, 0, 2))

        #Unpack `inputs` while handling the optional `e` parameter
        if len(inputs) == 3:
            u, y, e = inputs
        else:
            u, y = inputs
            e = None

        def scan_forward_pass(u, _):
            out_pred = vmap(self.forward_pass, (None, 0, 0, 0))(params, u, y, e)
            
            return out_pred, out_pred
        
        _, all_pred = lax.scan(scan_forward_pass, u, None, length = autoreg_steps)


        sum_loss = np.mean(np.abs((outputs[0:autoreg_steps, :, :] - all_pred))**loss_p)
        
        return sum_loss
 
    

    # update step when pre and refined training
    @partial(jit, static_argnums=(0, 4, 5, 6))
    def param_update(self, iteration, opt_state, batch, phase, training_param_tuple = None, autoreg_steps = None):
        """
        Update model parameters based on the optimization state and current batch.
        The behavior changes based on the training phase: 'PreTrain', 'RefineTrain'.
        """
        params = self.get_params(opt_state)

        # Select the gradient function based on the training phase
        if phase == 'PreTrain':
            gradients = grad(self.value_loss)(params, batch, training_param_tuple)
        elif phase == 'RefineTrain':
            gradients = grad(self.refine_loss)(params, batch, training_param_tuple, autoreg_steps)
        else:
            raise ValueError("Invalid Training phase. ")
            
        next_opt_state = self.opt_update(iteration, gradients, opt_state)
        
        return next_opt_state

    
    # update step when training with rates
    @partial(jit, static_argnums=(0, 5))
    def rates_param_update(self, iteration, opt_state, value_batch, rates_batch, training_param_tuple = None):
        """
        'Only for rates training"
        Update model parameters based on the optimization state and current batch.
        
        """
        params = self.get_params(opt_state)

        gradients = grad(self.rates_loss)(params, value_batch, rates_batch, training_param_tuple)
                   
        next_opt_state = self.opt_update(iteration, gradients, opt_state)
        
        return next_opt_state


    
    # training step for pre and refined
    def train(self, dataset, total_iterations = 100000, phase = 'PreTrain', training_param_dict = None, autoreg_steps = None):
        """Train the model with the given dataset, adapting the process based on the training phase."""
        
        data_iter = iter(dataset)
        progress_bar = trange(total_iterations)
        
        training_param_tuple = tuple(training_param_dict.values()) # Tuple can go inside jit
        (loss_p, w_rates) = training_param_tuple
        
        for iteration in progress_bar:
            
            batch = next(data_iter)
            
            self.opt_state = self.param_update(next(self.itercount), self.opt_state, batch, phase, training_param_tuple, autoreg_steps)

            # Logging and progress update every 10 iterations
            if iteration % 100 == 0:
                params = self.get_params(self.opt_state)
                if phase == 'PreTrain':
                    loss_value = self.value_loss(params, batch, training_param_tuple)
                    self.loss_log.append(loss_value)
                    progress_dict = {'Loss': loss_value}
                    progress_bar.set_postfix(progress_dict)  
                elif phase == 'RefineTrain':
                    loss_value = self.refine_loss(params, batch, training_param_tuple, autoreg_steps)
                    self.loss_log_ref.append(loss_value)  
                    progress_bar.set_postfix({'Refined Loss': loss_value})

    

    # reaction rates training step
    def rates_train(self, value_dataset, rates_dataset, total_iterations = 100000, training_param_dict = None):
        """Train the model with the pre training dataset and reaction rates dataset"""
        
        vdata_iter = iter(value_dataset)
        rdata_iter = iter(rates_dataset)
        progress_bar = trange(total_iterations)
        
        training_param_tuple = tuple(training_param_dict.values()) # Tuple can go inside jit
        (loss_p, w_rates) = training_param_tuple
        
        for iteration in progress_bar:
            
            value_batch = next(vdata_iter)
            rates_batch = next(rdata_iter)
            
            self.opt_state = self.rates_param_update(next(self.itercount), self.opt_state, value_batch, rates_batch, training_param_tuple)

            # Logging and progress update every 10 iterations
            if iteration % 100 == 0:
                params = self.get_params(self.opt_state)
                
                loss_value = self.rates_loss(params, value_batch, rates_batch, training_param_tuple)  
                self.loss_rates_log.append(loss_value)
                progress_dict = {'Total Rates Loss': loss_value}
                
                loss_value = w_rates*self.only_rates_loss(params, rates_batch, training_param_tuple)  
                self.loss_onlyrates_log.append(loss_value)
                progress_dict['Only Rates Loss'] = loss_value
                
                loss_value = self.value_loss(params, value_batch, training_param_tuple)  
                self.loss_valuerates_log.append(loss_value)
                progress_dict['Value Rates Loss'] = loss_value
                
                progress_bar.set_postfix(progress_dict) 
                
                  
          
       
           
    # predictions from DeepONet
    @partial(jit, static_argnums=(0,))
    def predict_deeponet(self, params, U, Y, E=None):
        
        g_pred = self.forward_pass(params, U, Y, E)
        
        return g_pred





















""" Mixture of experts model with DeepONets """

class MOEDeepONetModel:
    """A Mixture of Experts (MOE) model using multiple DeepONets with a gating network."""

    def __init__(self, branch_layers, trunk_layers, gating_layers, optimizer, activation=np.tanh, rng_key=random.PRNGKey(0)):
        """Initializes the MOE DeepONet model with separate DeepONets and a gating network."""
        self.rng_key = rng_key
        key_1, key_2, key_3, key_gating = random.split(self.rng_key, 4)

        # Instantiate each DeepONet with a unique RNG key
        self.first_DeepONet = DeepONet(branch_layers, trunk_layers, activation, key_1)
        self.second_DeepONet = DeepONet(branch_layers, trunk_layers, activation, key_2)
        self.third_DeepONet = DeepONet(branch_layers, trunk_layers, activation, key_3)

        # Instantiate the gating network
        self.gating_net = gatingMLP(gating_layers, key_gating, activation=activation)

        # Initialize optimizer with aggregated parameters from each network
        aggregated_params = (self.first_DeepONet.params, self.second_DeepONet.params, 
                             self.third_DeepONet.params, self.gating_net.params)
        
        self.opt_init, self.opt_update, self.get_params = optimizer
        self.opt_state = self.opt_init(aggregated_params)

        # Logging utilities
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_log_ref = []

        

    # MOE_Predict  
    def MOE_Predict(self, params, u, y):
        """Makes predictions using the MOE model by weighting DeepONet outputs with the gating network."""
        first_params, second_params, third_params, gating_params = params
        
        # Predict with each DeepONet
        G1 = self.first_DeepONet.predict(first_params, u, y)
        G2 = self.second_DeepONet.predict(second_params, u, y)
        G3 = self.third_DeepONet.predict(third_params, u, y)

        # Predict with the gating network
        Pk = self.gating_net.predict(gating_params, u)  

        # Weighted sum of DeepONet outputs
        g_out = G1*Pk[0] + G2*Pk[1] + G3*Pk[2]

        return g_out, Pk

    
    # MOE Forward pass       
    def forward_pass(self, params, u, y):
        """Computes forward pass through each DeepONet and the gating network, returning individual outputs."""
        first_params, second_params, third_params, gating_params = params
        
        # Individual predictions from each DeepONet
        G1 = self.first_DeepONet.predict(first_params, u, y)
        G2 = self.second_DeepONet.predict(second_params, u, y)
        G3 = self.third_DeepONet.predict(third_params, u, y)

        # Prediction from the gating network
        Pk = self.gating_net.predict(gating_params, u)

        return G1, G2, G3, Pk



    # MOE Loss based on model predictions and ground truth values
    # The loss is based on the original work on Mixture of Experts
    def value_loss(self, params, batch, training_param_tuple):
        """
        Computes the loss for a batch of data based on model predictions and ground truth values.
        """

        (loss_p, w_reg, beta, beta_s, beta_d, isExpert_Reg) = training_param_tuple
        
        inputs, outputs = batch
        u, y, _ = inputs  # e is not used directly in loss computation
        
        # Compute predictions for each DeepONet within the MOE architecture
        G1, G2, G3, Pk = vmap(self.forward_pass, (None, 0, 0))(params, u, y)
        
        # Calculate error vector for each DeepONet's predictions
        e1 = np.mean(np.abs(outputs - G1)**loss_p, axis=1)
        e2 = np.mean(np.abs(outputs - G2)**loss_p, axis=1)
        e3 = np.mean(np.abs(outputs - G3)**loss_p, axis=1)

        # Aggregate errors with gating network probabilities
        err_vec = beta * np.stack([-e1, -e2, -e3]).T
        
        # Compute softmax loss with logsumexp 
        loss_vec = -logsumexp(err_vec, axis=1, b=Pk) / (2 * beta)
        
        # Calculate mean loss across the batch
        loss = np.mean(loss_vec)
        
        return loss

    
    # Regularizer Loss for equal contribution from all the experts
    # From "Improving Expert Specialization in Mixture of Experts", Krishnamurthy et al. 2023
    def regularizer_loss(self, params, batch, training_param_tuple):
        
        """
        Computes the regularizer loss to encourage equal contribution from all experts.
        
        Parameters:
        - params: Parameters of the MOE DeepONet model.
        - batch: A batch of data including inputs and outputs.
        
        Returns:
        - The computed regularizer loss.
        
        """
        (loss_p, w_reg, beta, beta_s, beta_d, isExpert_Reg) = training_param_tuple
        
        inputs, outputs = batch
        u, y, e = inputs 
        
        _, _, _, Pk = vmap(self.forward_pass, (None, 0, 0))(params, u, y)

        N = u.shape[0] # Number of samples
        
        u_feat = u[:,[0, 1, 9]] # Select features from input for computing similarity/dissimilarity
        
        norm_sq = np.sum(u_feat ** 2, axis=1)
        diff_squared = norm_sq[:, None] + norm_sq[None, :] - 2 * np.dot(u_feat, u_feat.T)


        # Initialize S and D matrices.
        S = np.zeros((N, N))
        D = np.zeros((N, N))
    
        # Calculate the similarity and dissimilarity for each expert, then sum them up.
        for e in range(3):
            p_e = Pk[:, e][:, None]  # Probability of selecting expert e for each sample.
            # Compute similarity matrix for expert e.
            S_e = p_e * p_e.T * diff_squared
            S += beta_s * S_e
    
            # Compute dissimilarity matrix for expert e. We use broadcasting here.
            for e_prime in range(3):
                if e != e_prime:
                    p_e_prime = Pk[:, e_prime][None, :]  # Probability of selecting expert e' for each sample.
                    # Compute dissimilarity matrix for expert pair (e, e').
                    D_ee_prime = p_e * p_e_prime.T * diff_squared
                    D += beta_d * D_ee_prime
    
        # Subtract the diagonal elements which represent self-similarity/dissimilarity.
        S_diag = np.diag(S)
        D_diag = np.diag(D)
        
        # Compute the regularization term.
        L_s = (np.sum(S - np.diag(S_diag))/3 - np.sum(D - np.diag(D_diag))/6) / (N * (N - 1))
        
        return L_s

    
    
    # Definition of Total Loss
    def loss(self, params, batch, training_param_tuple):
        """
        Computes the total loss.
        
        """
        (loss_p, w_reg, beta, beta_s, beta_d, isExpert_Reg) = training_param_tuple
        # Compute value loss
        tot_loss = self.value_loss(params, batch, training_param_tuple)
        
        # Compute regularizer loss and adding to the total loss
        # Only if isExpert_Reg is True
        if isExpert_Reg:
            reg_loss = self.regularizer_loss(params, batch, training_param_tuple)
            tot_loss = tot_loss + w_reg*reg_loss
        
        
        return tot_loss



    def refine_loss(self, params, batch, training_param_tuple, autoreg_steps):
        """
        Computes the refinement loss over a given number of autoregressive steps.
        
        Parameters:
        - params: The parameters of the MOE DeepONet model.
        - batch: A batch of data including inputs and expected outputs.
        - autoreg_steps: The number of autoregressive steps to predict into the future.
        
        Returns:
        - The computed refinement loss.
        """
        (loss_p, w_reg, beta, beta_s, beta_d, isExpert_Reg) = training_param_tuple
        
        inputs, outputs = batch
        # Transpose outputs for batch processing
        outputs = np.transpose(outputs, (1, 0, 2))

        u, y, e = inputs
        e_u = e  # Placeholder if needed for future use

        def scan_forward_pass(u, _):
            # Forward pass through the network
            G1, G2, G3, Pk = vmap(self.forward_pass, (None, 0, 0))(params, u, y)
            # Combine predictions from all experts
            out_pred = G1*Pk[:, [0]] + G2*Pk[:, [1]] + G3*Pk[:, [2]]
            return out_pred, (G1, G2, G3, Pk)

        # Perform a scan over the autoregressive steps
        _, (st_G1, st_G2, st_G3, st_Pk) = lax.scan(scan_forward_pass, u, None, length=autoreg_steps)

        # Calculate the loss for each expert and aggregate
        e1 = np.mean(np.abs(outputs[0:autoreg_steps, :, :] - st_G1)**loss_p, axis=2)
        e2 = np.mean(np.abs(outputs[0:autoreg_steps, :, :] - st_G2)**loss_p, axis=2)
        e3 = np.mean(np.abs(outputs[0:autoreg_steps, :, :] - st_G3)**loss_p, axis=2)

        err_vec = beta * np.stack([-e1, -e2, -e3], axis=2)
        
        # # Compute softmax loss with logsumexp
        loss_vec = -logsumexp(err_vec, axis=2, b=st_Pk) / (2 * beta)

        sum_loss = np.mean(loss_vec)
        
        return sum_loss
 
    

    
    @partial(jit, static_argnums=(0, 4, 5, 6))
    def param_update(self, iteration, opt_state, batch, phase, training_param_tuple = None, autoreg_steps = None):
        """
        Update model parameters based on the optimization state and current batch.
        The behavior changes based on the training phase: 'PreTrain', 'RefineTrain', or 'RatesTrain'.
        """
        params = self.get_params(opt_state)

        # Select the gradient function based on the training phase
        if phase == 'PreTrain':
            gradients = grad(self.loss)(params, batch, training_param_tuple)
        elif phase == 'RefineTrain':
            gradients = grad(self.refine_loss)(params, batch, training_param_tuple, autoreg_steps)
        elif phase == 'RatesTrain':
            gradients = grad(self.rates_loss)(params, batch)
        else:
            raise ValueError("Invalid Training phase. ")
            
        next_opt_state = self.opt_update(iteration, gradients, opt_state)
        
        return next_opt_state


    

    def train(self, dataset, total_iterations=100000, phase='PreTrain', training_param_dict = None, autoreg_steps = None):
        """Train the model with the given dataset, adapting the process based on the training phase."""
        
        data_iter = iter(dataset)
        progress_bar = trange(total_iterations)
        
        training_param_tuple = tuple(training_param_dict.values()) # Tuple can go inside jit
        (loss_p, w_reg, beta, beta_s, beta_d, isExpert_Reg) = training_param_tuple
        
        for iteration in progress_bar:
            
            batch = next(data_iter)
            
            self.opt_state = self.param_update(next(self.itercount), self.opt_state, batch, phase, training_param_tuple, autoreg_steps)

            # Logging and progress update every 10 iterations
            if iteration % 10 == 0:
                params = self.get_params(self.opt_state)
                if phase == 'PreTrain':
                    loss_value = self.value_loss(params, batch, training_param_tuple)
                    self.loss_log.append(loss_value)
                    progress_dict = {'Loss': loss_value}
                    if isExpert_Reg:
                        reg_loss = self.regularizer_loss(params, batch, training_param_tuple)
                        progress_dict['Regularizer Loss'] =  reg_loss
                    progress_bar.set_postfix(progress_dict)  
                elif phase == 'RefineTrain':
                    loss_value = self.refine_loss(params, batch, training_param_tuple, autoreg_steps)
                    self.loss_log_ref.append(loss_value)  
                    progress_bar.set_postfix({'Refined Loss': loss_value})
                elif phase == 'RatesTrain':
                    loss_value = self.rates_loss(params, batch)  
                    self.loss_log_rates.append(loss_value)  
                    progress_bar.set_postfix({'Rates Loss': loss_value}) 
                
                  
          
       
           
    # predictions from MOE DeepONet
    @partial(jit, static_argnums=(0,))
    def predict_moe(self, params, U, Y, E):
        
        g_pred, Pk = self.MOE_Predict(params, U, Y)
        
        return g_pred, Pk























class ANNModel:
    """ Simple ANN model with loss formulations and training functionalities """

    def __init__(self, ann_layers, optimizer, activation=np.tanh, rng_key=random.PRNGKey(0)):
        
        """Initializes ANN model"""
        
        self.rng_key = rng_key

        # Instantiate of ANN with a unique RNG key
        self.model_ANN = modifiedMLP(ann_layers, rng_key, activation)

        # Initialize optimizer with parameters
        aggregated_params = self.model_ANN.params
        
        self.opt_init, self.opt_update, self.get_params = optimizer
        self.opt_state = self.opt_init(aggregated_params)

        # Logging utilities
        self.itercount = itertools.count()
        self.loss_log = []

        
    
    # ANN Forward pass       
    def forward_pass(self, params, x):
        """Computes forward pass through ANN """
        
        y = self.model_ANN.predict(params, x)

        return y


    
    # Comparing point wise predictions
    def value_loss(self, params, batch, training_param_tuple):
        """
        Computes the loss for a batch of data based on model predictions and ground truth values.
        """

        (loss_p, w_reg) = training_param_tuple
        
        x_batch, y_batch = batch
        
        out_pred = vmap(self.forward_pass, (None, 0))(params, x_batch)

        loss = np.mean(np.abs((y_batch - out_pred))**loss_p)
        
        return loss

    
    # Output Jacobian wrt to input of Trunk Net
    # Can be used for Physics Informed ML
    def out_grad(self, params, x):

        """
        ** NOT PROPERLY IMPLEMENTED **
        
        Evaluates output Jacobian wrt to one of the input
        
        """
        
        def temp_out(params, x):
            g_out= self.forward_pass(params, x)
            t_out= g_out[0]
            
            return t_out

        # Gives full Jacobian matrix
        out_jac = jacrev(lambda x: self.forward_pass(params, x))(x)

        # Only Temp Grad can be obtained like this as well
        dTdt=  grad(temp_out, argnums=2)(params, x)

        # Another way to obtain Temp Grad
        #dTdt= out_jac[0]
        
        return dTdt

    
    # Definition of Total Loss
    def total_loss(self, params, value_batch, training_param_tuple):
        """
        Computes the total loss.
        Value_loss + PiNN_loss
        
        """
        (loss_p, w_reg) = training_param_tuple
        
        # Compute value loss
        tot_loss = self.value_loss(params, value_batch, training_param_tuple)
               
        tot_loss = tot_loss #+ w_reg*pinn_loss
        
        return tot_loss
 
    

    # update step for model parameters
    @partial(jit, static_argnums=(0, 4))
    def param_update(self, iteration, opt_state, batch, training_param_tuple = None):
        """
        Update model parameters based on the optimization state and current batch.
        """
        params = self.get_params(opt_state)

        gradients = grad(self.total_loss)(params, batch, training_param_tuple)
            
        next_opt_state = self.opt_update(iteration, gradients, opt_state)
        
        return next_opt_state

    
    # training step
    def train(self, dataset, total_iterations = 100000, training_param_dict = None):
        """Train the model with the given dataset. """
        
        data_iter = iter(dataset)
        progress_bar = trange(total_iterations)
        
        training_param_tuple = tuple(training_param_dict.values()) # Tuple can go inside jit
        (loss_p, w_reg) = training_param_tuple
        
        for iteration in progress_bar:
            
            batch = next(data_iter)
            
            self.opt_state = self.param_update(next(self.itercount), self.opt_state, batch, training_param_tuple)

            # Logging and progress update every 10 iterations
            if iteration % 1000 == 0:
                params = self.get_params(self.opt_state)
                loss_value = self.value_loss(params, batch, training_param_tuple)
                self.loss_log.append(loss_value)
                progress_dict = {'Loss': loss_value}
                progress_bar.set_postfix(progress_dict)  

                    
                        
           
    # predictions from ANN
    @partial(jit, static_argnums=(0,))
    def predict_ann(self, params, X):
        
        Y = self.forward_pass(params, X)
        
        return Y

    