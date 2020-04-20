This repository contains code to run Constrained Bayesian Optimisation using the Expected Improvement acquisition function. 
I am using the code for an electrochemical project for which there are known constraints on the domain, such that the 
standard hypercube domain is undesirable. 

## Requirements
This code was implemented using Python 3.7.6 and the following packages:
- numpy
- matplotlib
- tensorflow
- gpflowopt

## Contact / Acknowledgements
If you use this code for your research, please cite or acknowledge the author (Penelope Jones, pj321@cam.ac.uk). 
Please feel free to contact me if you have any questions about this work.

I have based this work on the GPFlowOpt package [(GPflowOpt: A Bayesian Optimization Library using TensorFlow, 
N. Knudde, J. van der Herten, T. Dhaene, I. Couckuyt, 2017)](https://arxiv.org/abs/1711.03845).