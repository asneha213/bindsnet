import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import Network
from bindsnet.models import TwoLayerNetwork
from bindsnet.network.nodes import Input
from bindsnet.network.nodes import StochasticNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import Hebbian
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, \
                                       plot_performance, plot_voltages

import pdb


class SpikingAgent:                                       

    def __init__(self):

        self.num_inputs = 24
        self.num_outputs = 4
        self.time = 50
        self.dt = 1.0
        self.spikes = None

        # Build network.
        self.network = Network()
        X = Input(n=self.num_inputs, traces=True)
        Y = StochasticNodes(n=self.num_outputs, traces=True)
        self.network.add_layer(X, name='X')
        self.network.add_layer(Y, name='Y')

        self.w = 0.3 * torch.rand(self.num_inputs, self.num_outputs)
        self.connection = Connection(source=self.network.layers['X'], target=self.network.layers['Y'], w=self.w, update_rule=Hebbian, wmin=0.0, wmax=1.0, norm=78.4)
        self.network.add_connection(self.connection, source='X', target='Y')


        self.M1 = Monitor(obj=X, state_vars=['s'])
        self.M2 = Monitor(obj=Y, state_vars=['s'])
        self.vM = Monitor(self.network.layers['Y'], ['v'], time=self.time)


        self.network.add_monitor(monitor=self.M1, name='X')
        self.network.add_monitor(monitor=self.M2, name='Y')
        self.network.add_monitor(monitor=self.vM, name='voltage')

    def forward(self, state):

        #self.network.layers['Y'].v = torch.zeros(self.network.layers['Y'].shape)
        state_code = torch.ones(self.num_inputs)
        state_code[state] = 200

        # Lazily encode data as Poisson spike trains.
        spike_train = poisson(datum=state_code, time=self.time, dt=self.dt)
        #data_loader = poisson_loader(data=state, time=time, dt=dt)

        self.network.run_RL(inpts={'X' : spike_train}, time=self.time)
        self.spikes = {'X' : self.M1.get('s'), 'Y' : self.M2.get('s')}
        voltages = self.vM.get('v')

        firing_rates = torch.sum(self.spikes['Y'][0:self.num_outputs, -1*self.time:], dim=1)
        #print(firing_rates)

        return firing_rates.detach().numpy()

    def update_weights(self, tderror, state, action, alpha):
        spikes_input = self.spikes['X'][0:self.num_inputs, -1*self.time:]
        spikes_output = self.spikes['Y'][0:self.num_outputs, -1*self.time:]
        self.network.run_synaptic_updates(tderror, state, action, [spikes_input, spikes_output], alpha)


if __name__ == "__main__":
    spagent = SpikingAgent()
    firing_rates = spagent.forward()
    print(firing_rates)


