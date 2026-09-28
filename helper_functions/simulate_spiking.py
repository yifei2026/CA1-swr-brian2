__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""File contains functions to simulate the spiking network"""

from brian2 import *
import numpy as np
import random
import os
import sys
import os
sys.path.append(os.path.dirname(__file__ ) + '/../')

from scipy.signal import butter, filtfilt, welch
from run_spiking import path_folder
from helper_functions.utils_spiking import create_connections, \
    average_firing_rate, \
    Connectivity, extract_conn_submatrix, \
    standard_neuronal_parameters_for_brian
matplotlib.use('Agg')  # non-interactive backend for direct figure generation
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal import windows
from brian2 import *
import numpy as np
import random
import os
import sys
import os
sys.path.append(os.path.dirname(__file__ ) + '/../')
from scipy.signal import find_peaks
from run_spiking import path_folder
from helper_functions.utils_spiking import create_connections, \
    average_firing_rate, \
    Connectivity, extract_conn_submatrix, \
    standard_neuronal_parameters_for_brian,create_butter_bandpass
from helper_functions.plot_spiking_results import plot_extended_syn_depression

def simulate_1(filename, time_with_curr=0.01*second, my_width=3*ms,
                                    compress_step=10,figure_name=None):
    """Save simulation where you combine the effect of synaptic depression and current injection to switch between SWR
    and non-SWR states

    :param filename: str
        Name of spiking filename
    :param sigma_P: Brian amp
        Maximal absolute value of current injected to P cells
    :param time_with_curr: Brian second
            Length of current stimulation
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param fraction_stim: float, [0,1]
        Decides fraction of cells to be stimulated
    """

    warm_up_time = 1.5 * second
    simtime_current = time_with_curr * 2. + warm_up_time + 2 * second

    filename_full = os.path.join(path_folder, filename + '_step4.npz')

    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    for dic_name in ['dic_PP', 'dic_BP', 'dic_BB', 'dic_PA', 'dic_AP', 'dic_AA', 'dic_BA', 'dic_PB', 'dic_AB']:

        dic = info_dictionary[dic_name].item()
        c_aux = Connectivity(dic['g_update'], dic['prob_of_connect'],
                             dic['size_pre'], dic['size_post'],
                             dic['name'], dic['delay'])
        c_aux.rows_connect_matrix = dic['rows_connect_matrix']
        c_aux.cols_connect_matrix = dic['cols_connect_matrix']
        if dic_name == 'dic_PP':
            c_PP = c_aux
        if dic_name == 'dic_BP':
            c_BP = c_aux
        if dic_name == 'dic_BB':
            c_BB = c_aux
        if dic_name == 'dic_PA':
            c_PA = c_aux
        if dic_name == 'dic_AP':
            c_AP = c_aux
        if dic_name == 'dic_AA':
            c_AA = c_aux
        if dic_name == 'dic_PB':
            c_PB = c_aux
        if dic_name == 'dic_BA':
            c_BA = c_aux
        if dic_name == 'dic_AB':
            c_AB = c_aux

    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    rev_gabaB = network_params['rev_gabaB']
    rev_gabaA = network_params['rev_gabaA']
    tau_ampa = network_params['tau_ampa']
    tau_gabaB = network_params['tau_gabaB']
    tau_gabaA = network_params['tau_gabaA']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']

    eqs_neurons_current = '''
        dv/dt=(-g_leak*(v-v_rest)+input_ampa+input_gabaA+input_gabaB+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
        dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
        vthr_3d: volt
        extracurrent: amp
        trefr : second
        input_ampa = -g_ampa*(v-rev_ampa) : amp
        input_gabaA = -g_gabaA*(v-rev_gabaA): amp
        input_gabaB = -g_gabaB*(v-rev_gabaB) : amp
        '''

    # Initialize neuron group
    neurons = NeuronGroup(network_params['NP'] + network_params['NB'] + network_params['NA'],
                          model=eqs_neurons_current,
                          threshold='v > vthr_3d', reset='v=v_rest', refractory='trefr', method='euler')

    pop_p = neurons[:network_params['NP']]
    pop_b = neurons[network_params['NP']:network_params['NP'] + network_params['NB']]
    pop_a = neurons[network_params['NP'] + network_params['NB']:]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    neurons.extracurrent = 0. * pA

    # Connecting the network
    con_PP = Synapses(pop_p, pop_p, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

    con_AP = Synapses(pop_p, pop_a, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    con_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_AP.g_ap = c_AP.g_update

    con_AA = Synapses(pop_a, pop_a, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    con_AA.connect(i=c_AA.rows_connect_matrix, j=c_AA.cols_connect_matrix)
    con_AA.g_aa = c_AA.g_update

    con_PA = Synapses(pop_a, pop_p, 'g_pa : 1', on_pre='g_gabaA += g_pa*nS', delay=c_PA.delay)
    con_PA.connect(i=c_PA.rows_connect_matrix, j=c_PA.cols_connect_matrix)
    con_PA.g_pa = c_PA.g_update

    con_BP = Synapses(pop_p, pop_b, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b, pop_b, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update

    con_BA = Synapses(pop_a, pop_b, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    con_BA.connect(i=c_BA.rows_connect_matrix, j=c_BA.cols_connect_matrix)
    con_BA.g_ba = c_BA.g_update

    con_AB = Synapses(pop_b, pop_a, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_A']- v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))

    # Population firing rates
    FRP = PopulationRateMonitor(pop_p)  # PC
    FRB = PopulationRateMonitor(pop_b)  # PVBC
    FRA = PopulationRateMonitor(pop_a)  # OLM

    # Raster: only sample part of each population
    N_RASTER_PC = min(100, int(network_params['NP']))
    N_RASTER_PVBC = min(100, int(network_params['NB']))
    N_RASTER_OLM = min(100, int(network_params['NA']))

    spike_P = SpikeMonitor(pop_p[:N_RASTER_PC])
    spike_B = SpikeMonitor(pop_b[:N_RASTER_PVBC])
    spike_A = SpikeMonitor(pop_a[:N_RASTER_OLM])

    # Current monitors used for sharp-wave/ripple extraction
    b_input_to_p = StateMonitor(
        pop_p,
        'input_gabaB',
        record=np.random.randint(
            0, network_params['NP'], size=100
        )
    )
    monitor_extracurr = StateMonitor(
        pop_a, 'extracurrent', record=True
    )

    # - 1 - warm up, stays in non-SWR
    n_bist = Network(collect())
    n_bist.run(2*warm_up_time)


    t_array = np.asarray(FRP.t[::compress_step] / ms, dtype=float)

    rate_PC = np.asarray(
        FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )
    rate_PVBC = np.asarray(
        FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )
    rate_OLM = np.asarray(
        FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )

    extracurr_array = (
        np.mean(
            np.asarray(
                monitor_extracurr.extracurrent[:, ::compress_step]
            ),
            axis=0
        ) * 1e12
    )

    mean_B_input_p = (
        np.mean(
            np.asarray(
                b_input_to_p.input_gabaB[:, ::compress_step]
            ),
            axis=0
        ) * 1e12
    )

    fs = 1e4 / compress_step

    # Ripple: 90-180 Hz
    lowcut = 90
    highcut = 180
    b, a = create_butter_bandpass(
        lowcut, highcut, fs,
        order=2, btype='band'
    )
    membrane_potential = filtfilt(
        b, a, -mean_B_input_p
    )

    # Sharp wave: <5 Hz
    lowcut = -1.
    highcut = 5.
    b1, a1 = create_butter_bandpass(
        lowcut, highcut, fs,
        order=2, btype='low'
    )
    filtered_signal = filtfilt(
        b1, a1, -mean_B_input_p
    )

    peaks, _ = find_peaks(
        filtered_signal,
        height=30,
        distance=int(fs*0.1)
    )

    baseline = 0

    ripple_events, threshold, smoothed_ripple, max_duration = \
        detection_ripple_events(
            membrane_potential,
            fs=fs
        )

    SW_events = detect_sharp_wave_events(
        filtered_signal,
        peaks,
        baseline,
        fs,
        min_peak_amplitude=30,
        min_iei=100
    )

    print(SW_events)

    if SW_events is None or len(SW_events) == 0:
        print("警告：没有事件，跳过事件特性计算")
        fwhms, amplitudes, ieis, peak_times, SW_events, max_fwhms = \
            [0], [0], [0], [0], [], 0
    else:
        fwhms, amplitudes, ieis, peak_times, max_fwhms = \
            compute_sharp_wave_event_properties(
                filtered_signal,
                SW_events,
                baseline,
                fs,
                t_array
            )

    print(max_fwhms)

    if SW_events and ripple_events:
        swr_events = find_swr_from_sharpwaves(
            ripple_events,
            SW_events,
            fs
        )
        swr_count = len(swr_events)
        print(f"最终识别的SWR事件数量: {swr_count}")
    else:
        swr_events = []
        swr_count = 0
        print("没有检测到有效的SWR事件")

    # ===================== end of publication-style plotting block =====================
    plot_extended_syn_depression(
        filename=filename,
        output_folder=path_folder,
        t_array=t_array,
        rate_PC=rate_PC,
        rate_PVBC=rate_PVBC,
        rate_OLM=rate_OLM,
        filtered_signal=filtered_signal,
        membrane_potential=membrane_potential,
        spike_P_t=np.asarray(spike_P.t / ms, dtype=float),
        spike_P_i=np.asarray(spike_P.i, dtype=float),
        spike_B_t=np.asarray(spike_B.t / ms, dtype=float),
        spike_B_i=np.asarray(spike_B.i, dtype=float),
        spike_A_t=np.asarray(spike_A.t / ms, dtype=float),
        spike_A_i=np.asarray(spike_A.i, dtype=float),
        n_raster_pc=N_RASTER_PC,
        n_raster_pvbc=N_RASTER_PVBC,
        n_raster_olm=N_RASTER_OLM,
        output_name=figure_name,
        dpi=600
    )
    return (
        len(ripple_events),
        max_duration,
        len(SW_events),
        max_fwhms,
        swr_count
    )
def simulate_2(filename, sigma_P=200*pA, time_with_curr=0.01*second, my_width=3*ms,
                                    compress_step=10, fraction_stim=0.6,figure_name=None):
    """Save simulation where you combine the effect of synaptic depression and current injection to switch between SWR
    and non-SWR states

    :param filename: str
        Name of spiking filename
    :param sigma_P: Brian amp
        Maximal absolute value of current injected to P cells
    :param time_with_curr: Brian second
            Length of current stimulation
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param fraction_stim: float, [0,1]
        Decides fraction of cells to be stimulated
    """

    warm_up_time = 1.5 * second
    simtime_current = time_with_curr * 2. + warm_up_time + 2 * second

    filename_full = os.path.join(path_folder, filename + '_step4.npz')

    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    for dic_name in ['dic_PP', 'dic_BP', 'dic_BB', 'dic_PA', 'dic_AP', 'dic_AA', 'dic_BA', 'dic_PB', 'dic_AB']:

        dic = info_dictionary[dic_name].item()
        c_aux = Connectivity(dic['g_update'], dic['prob_of_connect'],
                             dic['size_pre'], dic['size_post'],
                             dic['name'], dic['delay'])
        c_aux.rows_connect_matrix = dic['rows_connect_matrix']
        c_aux.cols_connect_matrix = dic['cols_connect_matrix']
        if dic_name == 'dic_PP':
            c_PP = c_aux
        if dic_name == 'dic_BP':
            c_BP = c_aux
        if dic_name == 'dic_BB':
            c_BB = c_aux
        if dic_name == 'dic_PA':
            c_PA = c_aux
        if dic_name == 'dic_AP':
            c_AP = c_aux
        if dic_name == 'dic_AA':
            c_AA = c_aux
        if dic_name == 'dic_PB':
            c_PB = c_aux
        if dic_name == 'dic_BA':
            c_BA = c_aux
        if dic_name == 'dic_AB':
            c_AB = c_aux

    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    rev_gabaB = network_params['rev_gabaB']
    rev_gabaA = network_params['rev_gabaA']
    tau_ampa = network_params['tau_ampa']
    tau_gabaB = network_params['tau_gabaB']
    tau_gabaA = network_params['tau_gabaA']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']

    eqs_neurons_current = '''
        dv/dt=(-g_leak*(v-v_rest)+input_ampa+input_gabaA+input_gabaB+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
        dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
        vthr_3d: volt
        extracurrent: amp
        trefr : second
        input_ampa = -g_ampa*(v-rev_ampa) : amp
        input_gabaA = -g_gabaA*(v-rev_gabaA): amp
        input_gabaB = -g_gabaB*(v-rev_gabaB) : amp
        '''

    # Initialize neuron group
    neurons = NeuronGroup(network_params['NP'] + network_params['NB'] + network_params['NA'],
                          model=eqs_neurons_current,
                          threshold='v > vthr_3d', reset='v=v_rest', refractory='trefr', method='euler')

    pop_p = neurons[:network_params['NP']]
    pop_b = neurons[network_params['NP']:network_params['NP'] + network_params['NB']]
    pop_a = neurons[network_params['NP'] + network_params['NB']:]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    neurons.extracurrent = 0. * pA

    # Connecting the network
    con_PP = Synapses(pop_p, pop_p, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

    con_AP = Synapses(pop_p, pop_a, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    con_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_AP.g_ap = c_AP.g_update

    con_AA = Synapses(pop_a, pop_a, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    con_AA.connect(i=c_AA.rows_connect_matrix, j=c_AA.cols_connect_matrix)
    con_AA.g_aa = c_AA.g_update

    con_PA = Synapses(pop_a, pop_p, 'g_pa : 1', on_pre='g_gabaA += g_pa*nS', delay=c_PA.delay)
    con_PA.connect(i=c_PA.rows_connect_matrix, j=c_PA.cols_connect_matrix)
    con_PA.g_pa = c_PA.g_update

    con_BP = Synapses(pop_p, pop_b, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b, pop_b, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update

    con_BA = Synapses(pop_a, pop_b, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    con_BA.connect(i=c_BA.rows_connect_matrix, j=c_BA.cols_connect_matrix)
    con_BA.g_ba = c_BA.g_update

    con_AB = Synapses(pop_b, pop_a, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_A']- v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))

    # Population firing rates
    FRP = PopulationRateMonitor(pop_p)  # PC
    FRB = PopulationRateMonitor(pop_b)  # PVBC
    FRA = PopulationRateMonitor(pop_a)  # OLM

    # Raster: only sample part of each population
    N_RASTER_PC = min(100, int(network_params['NP']))
    N_RASTER_PVBC = min(100, int(network_params['NB']))
    N_RASTER_OLM = min(100, int(network_params['NA']))

    spike_P = SpikeMonitor(pop_p[:N_RASTER_PC])
    spike_B = SpikeMonitor(pop_b[:N_RASTER_PVBC])
    spike_A = SpikeMonitor(pop_a[:N_RASTER_OLM])

    # Current monitors used for sharp-wave/ripple extraction
    b_input_to_p = StateMonitor(
        pop_p,
        'input_gabaB',
        record=np.random.randint(
            0, network_params['NP'], size=100
        )
    )
    monitor_extracurr = StateMonitor(
        pop_a, 'extracurrent', record=True
    )

    # - 1 - warm up, stays in non-SWR
    n_bist = Network(collect())
    n_bist.run(warm_up_time)

    # - 2 - inject positive current to P
    idx_p = int(fraction_stim * network_params['NP'])
    pop_p[0:idx_p].extracurrent = sigma_P
    current_on_time = time_with_curr
    n_bist.run(current_on_time)

    # switch off current
    neurons.extracurrent = 0 * pA
    # - 3 - run in SWR state
    n_bist.run(2 * second)

    t_array = np.asarray(FRP.t[::compress_step] / ms, dtype=float)

    rate_PC = np.asarray(
        FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )
    rate_PVBC = np.asarray(
        FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )
    rate_OLM = np.asarray(
        FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )

    extracurr_array = (
        np.mean(
            np.asarray(
                monitor_extracurr.extracurrent[:, ::compress_step]
            ),
            axis=0
        ) * 1e12
    )

    mean_B_input_p = (
        np.mean(
            np.asarray(
                b_input_to_p.input_gabaB[:, ::compress_step]
            ),
            axis=0
        ) * 1e12
    )

    fs = 1e4 / compress_step

    # Ripple: 90-180 Hz
    lowcut = 90
    highcut = 180
    b, a = create_butter_bandpass(
        lowcut, highcut, fs,
        order=2, btype='band'
    )
    membrane_potential = filtfilt(
        b, a, -mean_B_input_p
    )

    # Sharp wave: <5 Hz
    lowcut = -1.
    highcut = 5.
    b1, a1 = create_butter_bandpass(
        lowcut, highcut, fs,
        order=2, btype='low'
    )
    filtered_signal = filtfilt(
        b1, a1, -mean_B_input_p
    )

    peaks, _ = find_peaks(
        filtered_signal,
        height=30,
        distance=int(fs*0.1)
    )

    baseline = 0

    ripple_events, threshold, smoothed_ripple, max_duration = \
        detection_ripple_events(
            membrane_potential,
            fs=fs
        )

    SW_events = detect_sharp_wave_events(
        filtered_signal,
        peaks,
        baseline,
        fs,
        min_peak_amplitude=30,
        min_iei=100
    )

    print(SW_events)

    if SW_events is None or len(SW_events) == 0:
        print("警告：没有事件，跳过事件特性计算")
        fwhms, amplitudes, ieis, peak_times, SW_events, max_fwhms = \
            [0], [0], [0], [0], [], 0
    else:
        fwhms, amplitudes, ieis, peak_times, max_fwhms = \
            compute_sharp_wave_event_properties(
                filtered_signal,
                SW_events,
                baseline,
                fs,
                t_array
            )

    print(max_fwhms)

    if SW_events and ripple_events:
        swr_events = find_swr_from_sharpwaves(
            ripple_events,
            SW_events,
            fs
        )
        swr_count = len(swr_events)
        print(f"最终识别的SWR事件数量: {swr_count}")
    else:
        swr_events = []
        swr_count = 0
        print("没有检测到有效的SWR事件")

    # ===================== end of publication-style plotting block =====================
    plot_extended_syn_depression(
        filename=filename,
        output_folder=path_folder,
        t_array=t_array,
        rate_PC=rate_PC,
        rate_PVBC=rate_PVBC,
        rate_OLM=rate_OLM,
        filtered_signal=filtered_signal,
        membrane_potential=membrane_potential,
        spike_P_t=np.asarray(spike_P.t / ms, dtype=float),
        spike_P_i=np.asarray(spike_P.i, dtype=float),
        spike_B_t=np.asarray(spike_B.t / ms, dtype=float),
        spike_B_i=np.asarray(spike_B.i, dtype=float),
        spike_A_t=np.asarray(spike_A.t / ms, dtype=float),
        spike_A_i=np.asarray(spike_A.i, dtype=float),
        n_raster_pc=N_RASTER_PC,
        n_raster_pvbc=N_RASTER_PVBC,
        n_raster_olm=N_RASTER_OLM,
        output_name=figure_name,
        dpi=600
    )
    return (
        len(ripple_events),
        max_duration,
        len(SW_events),
        max_fwhms,
        swr_count
    )

def simulate_3(filename, sigma_P=200*pA, time_with_curr=0.01*second, my_width=3*ms,
                                    compress_step=10, fraction_stim=0.6,figure_name=None):
    """Save simulation where you combine the effect of synaptic depression and current injection to switch between SWR
    and non-SWR states

    :param filename: str
        Name of spiking filename
    :param sigma_P: Brian amp
        Maximal absolute value of current injected to P cells
    :param time_with_curr: Brian second
            Length of current stimulation
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param fraction_stim: float, [0,1]
        Decides fraction of cells to be stimulated
    """

    warm_up_time = 1.5 * second
    simtime_current = time_with_curr * 2. + warm_up_time + 2 * second

    filename_full = os.path.join(path_folder, filename + '_step4.npz')

    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    for dic_name in ['dic_PP', 'dic_BP', 'dic_BB', 'dic_PA', 'dic_AP', 'dic_AA', 'dic_BA', 'dic_PB', 'dic_AB']:

        dic = info_dictionary[dic_name].item()
        c_aux = Connectivity(dic['g_update'], dic['prob_of_connect'],
                             dic['size_pre'], dic['size_post'],
                             dic['name'], dic['delay'])
        c_aux.rows_connect_matrix = dic['rows_connect_matrix']
        c_aux.cols_connect_matrix = dic['cols_connect_matrix']
        if dic_name == 'dic_PP':
            c_PP = c_aux
        if dic_name == 'dic_BP':
            c_BP = c_aux
        if dic_name == 'dic_BB':
            c_BB = c_aux
        if dic_name == 'dic_PA':
            c_PA = c_aux
        if dic_name == 'dic_AP':
            c_AP = c_aux
        if dic_name == 'dic_AA':
            c_AA = c_aux
        if dic_name == 'dic_PB':
            c_PB = c_aux
        if dic_name == 'dic_BA':
            c_BA = c_aux
        if dic_name == 'dic_AB':
            c_AB = c_aux

    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    rev_gabaB = network_params['rev_gabaB']
    rev_gabaA = network_params['rev_gabaA']
    tau_ampa = network_params['tau_ampa']
    tau_gabaB = network_params['tau_gabaB']
    tau_gabaA = network_params['tau_gabaA']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']

    eqs_neurons_current = '''
        dv/dt=(-g_leak*(v-v_rest)+input_ampa+input_gabaA+input_gabaB+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
        dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
        vthr_3d: volt
        extracurrent: amp
        trefr : second
        input_ampa = -g_ampa*(v-rev_ampa) : amp
        input_gabaA = -g_gabaA*(v-rev_gabaA): amp
        input_gabaB = -g_gabaB*(v-rev_gabaB) : amp
        '''

    # Initialize neuron group
    neurons = NeuronGroup(network_params['NP'] + network_params['NB'] + network_params['NA'],
                          model=eqs_neurons_current,
                          threshold='v > vthr_3d', reset='v=v_rest', refractory='trefr', method='euler')

    pop_p = neurons[:network_params['NP']]
    pop_b = neurons[network_params['NP']:network_params['NP'] + network_params['NB']]
    pop_a = neurons[network_params['NP'] + network_params['NB']:]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    neurons.extracurrent = 0. * pA

    # Connecting the network
    con_PP = Synapses(pop_p, pop_p, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

    con_AP = Synapses(pop_p, pop_a, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    con_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_AP.g_ap = c_AP.g_update

    con_AA = Synapses(pop_a, pop_a, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    con_AA.connect(i=c_AA.rows_connect_matrix, j=c_AA.cols_connect_matrix)
    con_AA.g_aa = c_AA.g_update

    con_PA = Synapses(pop_a, pop_p, 'g_pa : 1', on_pre='g_gabaA += g_pa*nS', delay=c_PA.delay)
    con_PA.connect(i=c_PA.rows_connect_matrix, j=c_PA.cols_connect_matrix)
    con_PA.g_pa = c_PA.g_update

    con_BP = Synapses(pop_p, pop_b, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b, pop_b, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update

    con_BA = Synapses(pop_a, pop_b, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    con_BA.connect(i=c_BA.rows_connect_matrix, j=c_BA.cols_connect_matrix)
    con_BA.g_ba = c_BA.g_update

    con_AB = Synapses(pop_b, pop_a, 'g_ab : 1', on_pre='g_gabaB += g_ab*nS', delay=c_AB.delay)
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_A']- v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))

    # Population firing rates
    FRP = PopulationRateMonitor(pop_p)  # PC
    FRB = PopulationRateMonitor(pop_b)  # PVBC
    FRA = PopulationRateMonitor(pop_a)  # OLM

    # Raster: only sample part of each population
    N_RASTER_PC = min(100, int(network_params['NP']))
    N_RASTER_PVBC = min(100, int(network_params['NB']))
    N_RASTER_OLM = min(100, int(network_params['NA']))

    spike_P = SpikeMonitor(pop_p[:N_RASTER_PC])
    spike_B = SpikeMonitor(pop_b[:N_RASTER_PVBC])
    spike_A = SpikeMonitor(pop_a[:N_RASTER_OLM])

    # Current monitors used for sharp-wave/ripple extraction
    b_input_to_p = StateMonitor(
        pop_p,
        'input_gabaB',
        record=np.random.randint(
            0, network_params['NP'], size=100
        )
    )
    monitor_extracurr = StateMonitor(
        pop_a, 'extracurrent', record=True
    )


    # - 1 - warm up, stays in non-SWR
    n_bist = Network(collect())
    n_bist.run(warm_up_time)

    # - 2 - inject positive current to P
    idx_p = int(fraction_stim * network_params['NP'])
    pop_p[0:idx_p].extracurrent = sigma_P
    current_on_time = time_with_curr
    n_bist.run(current_on_time)

    # switch off current
    neurons.extracurrent = 0 * pA
    # - 3 - run in SWR state
    n_bist.run(1 * second)

    idx_p = int(fraction_stim * network_params['NP'])
    pop_p[0:idx_p].extracurrent = -sigma_P
    current_on_time = time_with_curr
    n_bist.run(current_on_time)

    # switch off current
    neurons.extracurrent = 0 * pA
    # - 3 - run in SWR state
    n_bist.run(1 * second)

    t_array = np.asarray(FRP.t[::compress_step] / ms, dtype=float)

    rate_PC = np.asarray(
        FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )
    rate_PVBC = np.asarray(
        FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )
    rate_OLM = np.asarray(
        FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )

    extracurr_array = (
        np.mean(
            np.asarray(
                monitor_extracurr.extracurrent[:, ::compress_step]
            ),
            axis=0
        ) * 1e12
    )

    mean_B_input_p = (
        np.mean(
            np.asarray(
                b_input_to_p.input_gabaB[:, ::compress_step]
            ),
            axis=0
        ) * 1e12
    )

    fs = 1e4 / compress_step

    # Ripple: 90-180 Hz
    lowcut = 90
    highcut = 180
    b, a = create_butter_bandpass(
        lowcut, highcut, fs,
        order=2, btype='band'
    )
    membrane_potential = filtfilt(
        b, a, -mean_B_input_p
    )

    # Sharp wave: <5 Hz
    lowcut = -1.
    highcut = 5.
    b1, a1 = create_butter_bandpass(
        lowcut, highcut, fs,
        order=2, btype='low'
    )
    filtered_signal = filtfilt(
        b1, a1, -mean_B_input_p
    )

    peaks, _ = find_peaks(
        filtered_signal,
        height=30,
        distance=int(fs*0.1)
    )

    baseline = 0

    ripple_events, threshold, smoothed_ripple, max_duration = \
        detection_ripple_events(
            membrane_potential,
            fs=fs
        )

    SW_events = detect_sharp_wave_events(
        filtered_signal,
        peaks,
        baseline,
        fs,
        min_peak_amplitude=30,
        min_iei=100
    )

    print(SW_events)

    if SW_events is None or len(SW_events) == 0:
        print("警告：没有事件，跳过事件特性计算")
        fwhms, amplitudes, ieis, peak_times, SW_events, max_fwhms = \
            [0], [0], [0], [0], [], 0
    else:
        fwhms, amplitudes, ieis, peak_times, max_fwhms = \
            compute_sharp_wave_event_properties(
                filtered_signal,
                SW_events,
                baseline,
                fs,
                t_array
            )

    print(max_fwhms)

    if SW_events and ripple_events:
        swr_events = find_swr_from_sharpwaves(
            ripple_events,
            SW_events,
            fs
        )
        swr_count = len(swr_events)
        print(f"最终识别的SWR事件数量: {swr_count}")
    else:
        swr_events = []
        swr_count = 0
        print("没有检测到有效的SWR事件")

    # ===================== end of publication-style plotting block =====================
    plot_extended_syn_depression(
        filename=filename,
        output_folder=path_folder,
        t_array=t_array,
        rate_PC=rate_PC,
        rate_PVBC=rate_PVBC,
        rate_OLM=rate_OLM,
        filtered_signal=filtered_signal,
        membrane_potential=membrane_potential,
        spike_P_t=np.asarray(spike_P.t / ms, dtype=float),
        spike_P_i=np.asarray(spike_P.i, dtype=float),
        spike_B_t=np.asarray(spike_B.t / ms, dtype=float),
        spike_B_i=np.asarray(spike_B.i, dtype=float),
        spike_A_t=np.asarray(spike_A.t / ms, dtype=float),
        spike_A_i=np.asarray(spike_A.i, dtype=float),
        n_raster_pc=N_RASTER_PC,
        n_raster_pvbc=N_RASTER_PVBC,
        n_raster_olm=N_RASTER_OLM,
        output_name=figure_name,
        dpi=600
    )
    return (
        len(ripple_events),
        max_duration,
        len(SW_events),
        max_fwhms,
        swr_count
    )

def simulate_4(filename, sigma_P=200*pA, time_with_curr=0.01*second, my_width=3*ms,simtime_current=1.5 * second,
                                    compress_step=10, fraction_stim=0.6,figure_name=None):
    """Save simulation where you combine the effect of synaptic depression and current injection to switch between SWR
    and non-SWR states

    :param filename: str
        Name of spiking filename
    :param sigma_P: Brian amp
        Maximal absolute value of current injected to P cells
    :param time_with_curr: Brian second
            Length of current stimulation
    :param my_width: Brian second
        Width of Gaussian smoothing of instantaneous population firing rate
    :param compress_step: int
        Compression factor to store simulation results for later plotting
    :param fraction_stim: float, [0,1]
        Decides fraction of cells to be stimulated
    """

    warm_up_time = 1.5 * second
    #simtime_current = time_with_curr * 2. + warm_up_time + 2 * second

    filename_full = os.path.join(path_folder, filename + '_step6.npz')

    data = np.load(filename_full, encoding='latin1', allow_pickle=True)
    info_dictionary = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    network_params = info_dictionary['network_params'].item()

    for dic_name in ['dic_PP', 'dic_BP', 'dic_BB', 'dic_PA', 'dic_AP', 'dic_AA', 'dic_BA', 'dic_PB', 'dic_AB']:

        dic = info_dictionary[dic_name].item()
        c_aux = Connectivity(dic['g_update'], dic['prob_of_connect'],
                             dic['size_pre'], dic['size_post'],
                             dic['name'], dic['delay'])
        c_aux.rows_connect_matrix = dic['rows_connect_matrix']
        c_aux.cols_connect_matrix = dic['cols_connect_matrix']
        if dic_name == 'dic_PP':
            c_PP = c_aux
        if dic_name == 'dic_BP':
            c_BP = c_aux
        if dic_name == 'dic_BB':
            c_BB = c_aux
        if dic_name == 'dic_PA':
            c_PA = c_aux
        if dic_name == 'dic_AP':
            c_AP = c_aux
        if dic_name == 'dic_AA':
            c_AA = c_aux
        if dic_name == 'dic_PB':
            c_PB = c_aux
        if dic_name == 'dic_BA':
            c_BA = c_aux
        if dic_name == 'dic_AB':
            c_AB = c_aux

    g_leak = network_params['g_leak']
    v_rest = network_params['v_rest']
    rev_ampa = network_params['rev_ampa']
    rev_gabaB = network_params['rev_gabaB']
    rev_gabaA = network_params['rev_gabaA']
    tau_ampa = network_params['tau_ampa']
    tau_gabaB = network_params['tau_gabaB']
    tau_gabaA = network_params['tau_gabaA']
    bg_curr = network_params['bg_curr']
    mem_cap = network_params['mem_cap']

    eqs_neurons_current = '''
        dv/dt=(-g_leak*(v-v_rest)+input_ampa+input_gabaA+input_gabaB+bg_curr+extracurrent)/mem_cap : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gabaB/dt = -g_gabaB/tau_gabaB : siemens
        dg_gabaA/dt = -g_gabaA/tau_gabaA : siemens
        vthr_3d: volt
        extracurrent: amp
        trefr : second
        input_ampa = -g_ampa*(v-rev_ampa) : amp
        input_gabaA = -g_gabaA*(v-rev_gabaA): amp
        input_gabaB = -g_gabaB*(v-rev_gabaB) : amp
        '''

    # Initialize neuron group
    neurons = NeuronGroup(network_params['NP'] + network_params['NB'] + network_params['NA'],
                          model=eqs_neurons_current,
                          threshold='v > vthr_3d', reset='v=v_rest', refractory='trefr', method='euler')

    pop_p = neurons[:network_params['NP']]
    pop_b = neurons[network_params['NP']:network_params['NP'] + network_params['NB']]
    pop_a = neurons[network_params['NP'] + network_params['NB']:]

    pop_p.vthr_3d = network_params['vthr_P']
    pop_b.vthr_3d = network_params['vthr_B']
    pop_a.vthr_3d = network_params['vthr_A']

    pop_p.trefr = network_params['trefr_P']
    pop_b.trefr = network_params['trefr_B']
    pop_a.trefr = network_params['trefr_A']

    neurons.extracurrent = 0. * pA

    # Connecting the network
    con_PP = Synapses(pop_p, pop_p, 'g_pp : 1', on_pre='g_ampa += g_pp*nS', delay=c_PP.delay)
    con_PP.connect(i=c_PP.rows_connect_matrix, j=c_PP.cols_connect_matrix)
    con_PP.g_pp = c_PP.g_update

    con_AP = Synapses(pop_p, pop_a, 'g_ap : 1', on_pre='g_ampa += g_ap*nS', delay=c_AP.delay)
    con_AP.connect(i=c_AP.rows_connect_matrix, j=c_AP.cols_connect_matrix)
    con_AP.g_ap = c_AP.g_update

    con_AA = Synapses(pop_a, pop_a, 'g_aa: 1', on_pre='g_gabaA += g_aa*nS', delay=c_AA.delay)
    con_AA.connect(i=c_AA.rows_connect_matrix, j=c_AA.cols_connect_matrix)
    con_AA.g_aa = c_AA.g_update

    con_PA = Synapses(pop_a, pop_p, 'g_pa : 1', on_pre='g_gabaA += g_pa*nS', delay=c_PA.delay)
    con_PA.connect(i=c_PA.rows_connect_matrix, j=c_PA.cols_connect_matrix)
    con_PA.g_pa = c_PA.g_update

    con_BP = Synapses(pop_p, pop_b, 'g_bp : 1', on_pre='g_ampa += g_bp*nS', delay=c_BP.delay)
    con_BP.connect(i=c_BP.rows_connect_matrix, j=c_BP.cols_connect_matrix)
    con_BP.g_bp = c_BP.g_update

    con_BB = Synapses(pop_b, pop_b, 'g_bb : 1', on_pre='g_gabaB += g_bb*nS', delay=c_BB.delay)
    con_BB.connect(i=c_BB.rows_connect_matrix, j=c_BB.cols_connect_matrix)
    con_BB.g_bb = c_BB.g_update

    con_PB = Synapses(pop_b, pop_p, 'g_pb : 1', on_pre='g_gabaB += g_pb*nS', delay=c_PB.delay)
    con_PB.connect(i=c_PB.rows_connect_matrix, j=c_PB.cols_connect_matrix)
    con_PB.g_pb = c_PB.g_update

    con_BA = Synapses(pop_a, pop_b, 'g_ba : 1', on_pre='g_gabaA += g_ba*nS', delay=c_BA.delay)
    con_BA.connect(i=c_BA.rows_connect_matrix, j=c_BA.cols_connect_matrix)
    con_BA.g_ba = c_BA.g_update

    # Plastic B --> A
    # We use the clamp option to clamp x at intermediate value and see bistability in spontaneous network
    clamp_depr = 1
    eqs_std_AB = '''
       g_ab : 1
       dx / dt = clamp_depr * (1. - x) / tau_depr_AB : 1 (clock-driven)
       '''

    con_AB = Synapses(pop_b, pop_a, model=eqs_std_AB,
                      on_pre='''g_gabaB += x*g_ab*nS
                            x = clip(x - x * eta_ab, 0, 1)
                             ''', delay=c_AB.delay, method='exact')
    con_AB.connect(i=c_AB.rows_connect_matrix, j=c_AB.cols_connect_matrix)
    con_AB.g_ab = c_AB.g_update
    con_AB.x = 1
    tau_depr_AB = network_params['tau_depr_AB']

    # Set initial values to start from non-SWR state
    pop_b.v = v_rest
    pop_p.v = v_rest
    pop_a.v = v_rest + (network_params['vthr_A']- v_rest) * np.random.rand((network_params['NA']))
    neurons.g_ampa = 0.01 * nS * np.random.rand((network_params['NP'] + network_params['NB'] + network_params[
        'NA']))
    neurons.g_gabaA = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))
    neurons.g_gabaB = 0.01 * nS * np.random.rand(
        (network_params['NP'] + network_params['NB'] + network_params['NA']))

    # Population firing rates
    FRP = PopulationRateMonitor(pop_p)  # PC
    FRB = PopulationRateMonitor(pop_b)  # PVBC
    FRA = PopulationRateMonitor(pop_a)  # OLM

    # Raster: only sample part of each population
    N_RASTER_PC = min(100, int(network_params['NP']))
    N_RASTER_PVBC = min(100, int(network_params['NB']))
    N_RASTER_OLM = min(100, int(network_params['NA']))

    spike_P = SpikeMonitor(pop_p[:N_RASTER_PC])
    spike_B = SpikeMonitor(pop_b[:N_RASTER_PVBC])
    spike_A = SpikeMonitor(pop_a[:N_RASTER_OLM])

    # Current monitors used for sharp-wave/ripple extraction
    b_input_to_p = StateMonitor(
        pop_p,
        'input_gabaB',
        record=np.random.randint(
            0, network_params['NP'], size=100
        )
    )
    monitor_extracurr = StateMonitor(
        pop_a, 'extracurrent', record=True
    )

    curr_to_pop = 'P'
    eta_ab = 0.18

    # - 1 - warm up, stays in non-SWR
    n_bist = Network(collect())
    n_bist.run(simtime_current)

    # - 2 - inject positive current to P
    idx_p = int(fraction_stim * network_params['NP'])
    pop_p[0:idx_p].extracurrent = sigma_P
    current_on_time = time_with_curr
    n_bist.run(current_on_time)

    # switch off current
    neurons.extracurrent = 0 * pA
    # - 3 - run in SWR state
    n_bist.run(2 * second)

    t_array = np.asarray(FRP.t[::compress_step] / ms, dtype=float)

    rate_PC = np.asarray(
        FRP.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )
    rate_PVBC = np.asarray(
        FRB.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )
    rate_OLM = np.asarray(
        FRA.smooth_rate('gaussian', width=my_width)[::compress_step] / Hz,
        dtype=float
    )

    extracurr_array = (
        np.mean(
            np.asarray(
                monitor_extracurr.extracurrent[:, ::compress_step]
            ),
            axis=0
        ) * 1e12
    )

    mean_B_input_p = (
        np.mean(
            np.asarray(
                b_input_to_p.input_gabaB[:, ::compress_step]
            ),
            axis=0
        ) * 1e12
    )

    fs = 1e4 / compress_step

    # Ripple: 90-180 Hz
    lowcut = 90
    highcut = 180
    b, a = create_butter_bandpass(
        lowcut, highcut, fs,
        order=2, btype='band'
    )
    membrane_potential = filtfilt(
        b, a, -mean_B_input_p
    )

    # Sharp wave: <5 Hz
    lowcut = -1.
    highcut = 5.
    b1, a1 = create_butter_bandpass(
        lowcut, highcut, fs,
        order=2, btype='low'
    )
    filtered_signal = filtfilt(
        b1, a1, -mean_B_input_p
    )

    peaks, _ = find_peaks(
        filtered_signal,
        height=30,
        distance=int(fs*0.1)
    )

    baseline = 0

    ripple_events, threshold, smoothed_ripple, max_duration = \
        detection_ripple_events(
            membrane_potential,
            fs=fs
        )

    SW_events = detect_sharp_wave_events(
        filtered_signal,
        peaks,
        baseline,
        fs,
        min_peak_amplitude=30,
        min_iei=100
    )

    print(SW_events)

    if SW_events is None or len(SW_events) == 0:
        print("警告：没有事件，跳过事件特性计算")
        fwhms, amplitudes, ieis, peak_times, SW_events, max_fwhms = \
            [0], [0], [0], [0], [], 0
    else:
        fwhms, amplitudes, ieis, peak_times, max_fwhms = \
            compute_sharp_wave_event_properties(
                filtered_signal,
                SW_events,
                baseline,
                fs,
                t_array
            )

    print(max_fwhms)

    if SW_events and ripple_events:
        swr_events = find_swr_from_sharpwaves(
            ripple_events,
            SW_events,
            fs
        )
        swr_count = len(swr_events)
        print(f"最终识别的SWR事件数量: {swr_count}")
    else:
        swr_events = []
        swr_count = 0
        print("没有检测到有效的SWR事件")
    # Plotting is kept separate from the Brian2 simulation and analysis code.
    # All arrays below are the unchanged outputs calculated above.
    plot_extended_syn_depression(
        filename=filename,
        output_folder=path_folder,
        t_array=t_array,
        rate_PC=rate_PC,
        rate_PVBC=rate_PVBC,
        rate_OLM=rate_OLM,
        filtered_signal=filtered_signal,
        membrane_potential=membrane_potential,
        spike_P_t=np.asarray(spike_P.t / ms, dtype=float),
        spike_P_i=np.asarray(spike_P.i, dtype=float),
        spike_B_t=np.asarray(spike_B.t / ms, dtype=float),
        spike_B_i=np.asarray(spike_B.i, dtype=float),
        spike_A_t=np.asarray(spike_A.t / ms, dtype=float),
        spike_A_i=np.asarray(spike_A.i, dtype=float),
        n_raster_pc=N_RASTER_PC,
        n_raster_pvbc=N_RASTER_PVBC,
        n_raster_olm=N_RASTER_OLM,
        output_name=figure_name,
        dpi=600
    )
    return (
        len(ripple_events),
        max_duration,
        len(SW_events),
        max_fwhms,
        swr_count
    )




def detection_ripple_events(signal, fs=1000, gaussian_std=0.65, gaussian_window_ms=50, threshold_factor=2,
                            min_duration_ms=30, start_time_ms=0):
    """
    检测Ripple事件。

    参数：
        signal: 输入的信号（1D数组）。
        fs: 采样率 (Hz)，默认1000。
        gaussian_std: 高斯窗口的标准差，默认0.65。
        gaussian_window_ms: 高斯窗口的宽度 (ms)，默认50ms。
        threshold_factor: 检测阈值因子，默认中值 + 2倍标准差。
        min_duration_ms: 最短事件持续时间，默认30ms。
        start_time_ms: 开始检测的时间 (ms)，默认100ms。

    返回：
        ripple_events: 包含检测到的Ripple事件的起始时间和结束时间（以秒为单位）。
        threshold: 使用的检测阈值。
        smoothed_ripple: 平滑后的瞬时幅度信号。
    """
    # 转换开始时间为样本点
    start_sample = int((start_time_ms / 1000) * fs)

    # 截取从start_sample开始的信号
    truncated_signal = signal[start_sample:]

    # 1. 使用Hilbert变换计算信号的瞬时幅度
    analytic_signal = hilbert(truncated_signal)
    ripple_amplitude = np.abs(analytic_signal)

    # 2. 使用高斯窗口平滑瞬时幅度
    gaussian_window_samples = int((gaussian_window_ms / 1000) * fs)  # 高斯窗口的样本数
    gaussian_window = windows.gaussian(gaussian_window_samples, std=gaussian_std)
    smoothed_ripple = convolve(ripple_amplitude, gaussian_window / gaussian_window.sum(), mode='same')

    # 3. 计算检测阈值#
    ripple_median = np.median(smoothed_ripple)
    ripple_std = np.std(smoothed_ripple)
    threshold = ripple_median + threshold_factor * ripple_std

    # 4. 检测Ripple事件
    above_threshold = smoothed_ripple > threshold
    min_duration_samples = int((min_duration_ms / 1000) * fs)
    ripple_events = []
    start_idx = None

    for i in range(1, len(above_threshold)):
        if above_threshold[i] and not above_threshold[i - 1]:
            start_idx = i
        elif not above_threshold[i] and above_threshold[i - 1]:
            if start_idx is not None and i - start_idx >= min_duration_samples:
                # 转换回原始信号的时间索引
                event_start = (start_idx + start_sample) / fs
                event_end = (i + start_sample) / fs
                ripple_events.append((event_start, event_end))
            start_idx = None

    if start_idx is not None and len(above_threshold) - start_idx >= min_duration_samples:
        event_start = (start_idx + start_sample) / fs
        event_end = (len(above_threshold) + start_sample) / fs
        ripple_events.append((event_start, event_end))
    # 计算每个事件的持续时间
    durations = [event_end - event_start for event_start, event_end in ripple_events]
    # 获取最大持续时间
    if len(durations) == 0:
        max_duration = 0  # 或其他默认值
    else:
        max_duration = max(durations)*1000
    # 打印最大持续时间
    #print(max_duration)
    return ripple_events, threshold, smoothed_ripple,max_duration


def detect_sharp_wave_events(signal, peaks, baseline, fs, min_peak_amplitude=30, min_iei=100):

    events = []  # 存储事件边界和峰值索引

    # **第一步：检测峰值**，峰值不小于30 pA
    for peak in peaks:
        if signal[peak] < min_peak_amplitude:
            continue  # 跳过不符合条件的事件
            print("检测峰值小于30 pA，返回空列表")

        events.append(peak)  # 记录符合条件的峰值
    if not events:
        print("没有有效的SWR事件，返回空列表")
        return None
    # **第二步：检测峰值间隔**，确保事件之间的最小间隔不小于100毫秒
    valid_events = []
    prev_peak = None

    for peak in events:
        if prev_peak is not None:
            # 计算两个峰值之间的时间间隔，单位为毫秒
            peak_distance_ms = (peak - prev_peak) / fs * 1000  # 转换为毫秒

            if peak_distance_ms >= min_iei:
                valid_events.append(peak)
            else:
                print("检测两个峰值之间的时间间隔，返回空列表")
                continue



        # 添加第一个峰值到有效事件列表
        if prev_peak is None:
            valid_events.append(peak)

        prev_peak = peak
    if not valid_events:
        print("没有有效的SWR事件，返回空列表")
        return None

    # **第三步：计算事件的持续时间**，确保每个事件的持续时间小于150毫秒
    final_valid_events = []
    for peak in valid_events:
        # 找到每个Sharp Wave事件的开始时间和结束时间
        half_max = baseline + (signal[peak] - baseline) / 2

        # 计算事件的开始时间（信号第一次达到半最大值的位置）
        left_idx = peak
        while left_idx > 0 and signal[left_idx] > half_max:
            left_idx -= 1

        # 计算事件的结束时间（信号第二次达到半最大值的位置）
        right_idx = peak
        while right_idx < len(signal) - 1 and signal[right_idx] > half_max:
            right_idx += 1

        # 计算事件的持续时间（单位毫秒）
        duration = (right_idx - left_idx) / fs * 1000  # 计算事件持续时间，单位毫秒

        # 只有持续时间小于max_duration的事件才有效
        if duration <= 150:
            final_valid_events.append((left_idx, right_idx, peak, duration))  # 存储有效事件的信息
        else:
            print("检测持续时间大于max_duration，返回空列表")
            continue

    return final_valid_events

def compute_sharp_wave_event_properties(signal, events, baseline, fs, t_array):
    fwhms = []
    amplitudes = []
    peak_times = []
    ieis = []  # 用于存储事件间隔（IEI）
    for i, event in enumerate(events):
        left_idx, right_idx, peak_idx, duration = event
        # 计算 FWHM（单位为秒）
        fwhms.append(duration)
        # 计算幅度（峰值减去基线）
        amplitude = signal[peak_idx] - baseline
        amplitudes.append(amplitude)
        # 记录事件峰值时间
        peak_times.append(t_array[peak_idx])
        # 计算事件间隔（IEI）
        # 计算事件间隔（IEI）
        if i > 0:  # 从第二个事件开始计算
            prev_right_idx = events[i - 1][1]  # 上一个事件的结束时间（right_idx）
            iei = (left_idx - prev_right_idx) / fs  # 计算IEI，单位为秒
            ieis.append(iei)
    # 获取最大持续时间
    if len(fwhms) == 0:
        max_duration = 0  # 或其他默认值
    else:
        max_duration = max(fwhms)
    return fwhms, amplitudes, ieis, peak_times,max_duration
def find_swr_from_sharpwaves(ripple_events, sharp_wave_events, fs):
    """
    判断每个 Sharp wave 是否与至少一个 Ripple 重叠，从而定义为一个 SWR 事件。

    参数：
        ripple_events: list of (ripple_start_time, ripple_end_time)，单位为秒
        sharp_wave_events: list of (left_idx, right_idx, peak_idx, duration)，索引单位
        fs: 采样率

    返回：
        swr_events: list of (sharp_start_time, sharp_end_time, peak_time)，每个是一个SWR事件
    """
    swr_events = []
    for left_idx, right_idx, peak_idx, duration in sharp_wave_events:
        sharp_start = left_idx / fs
        sharp_end = right_idx / fs
        peak_time = peak_idx / fs

        # 判断是否有至少一个ripple事件与当前sharp wave重叠
        for ripple_start, ripple_end in ripple_events:
            if ripple_start <= sharp_end and ripple_end >= sharp_start:
                swr_events.append((sharp_start, sharp_end, peak_time))
                break
    return swr_events
