## 2.5 System Identification of Nonlinear Dynamical Systems 

### Objective 
Perform system identification (i.e., learning the model of a dynamical system) of challenging nonlinear systems.

### Dataset
Cortical Responses Evoked by Wrist Joint Manipulation

### Experimental Plan
* **Phase 1**: Train different kinds of models: NNARX networks, simple recurrent neural networks (RNNs) and LSTM RNNs (optionally, GRU RNNS can also be considered). 
* **Phase 2**: Explore different architectures (number of states, number of layers, etc.) for each kind of model. 
* **Phase 3**: Compare the obtained results in terms of: model accuracy in simulation; time required to perform the training; model complexity: number of parameters and FLOPS to perform the output sample prediction. 
