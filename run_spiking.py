__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains all instruction to create spiking network and reproduce related figures in the manuscript.

Note: randomness is involved in network creation, so there might be some deviation from the values reported in the
manuscript."""

import sys
import os
sys.path.append(os.path.dirname( __file__ ) + '/../')

# Store all simulations and plots
if not os.path.exists(os.path.join(os.path.dirname( __file__ ), 'results')):
    os.makedirs(os.path.join(os.path.dirname( __file__ ), 'results'))
path_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
from brian2 import *


from helper_functions.construct_spiking_network import *
from helper_functions.simulate_spiking import *
filename_spiking = 'swr_slice_network'


if __name__ == '__main__':

    seed(3)  # Brian seed

    # =============== 1 - Construct the network ========== #
    print('Constructing the spiking network..')
    '''step1_PA(filename_spiking, make_plots=False)

    step2_PB(filename_spiking, make_plots=False)

    step3_add_Wab(filename_spiking, make_plots=False)
    step4_merge_subnetworks(filename_spiking, make_plots=False)
    step5_current_injection(filename_spiking, curr_to_pop='P')  # -> this is more of a check up, but can be skipped
    step6_spontaneous(filename_spiking)'''

    # ================ 2 - Simulations and figures default spiking network ========== #

    # FIG 2
    print('Simulating network for Fig. 2')
    # simulate_extended_syn_depression performs the unchanged Brian2
    # simulation/analysis and then calls plot_extended_syn_depression().
    simulate_1(filename_spiking, figure_name='Fig1')
    simulate_2(filename_spiking, sigma_P=200 * pA, fraction_stim=1., figure_name='Fig2')
    simulate_3(filename_spiking, sigma_P=200 * pA, fraction_stim=1., figure_name='Fig3')
    simulate_4(filename_spiking, sigma_P=200 * pA, fraction_stim=1., figure_name='Fig4')
