"""Plotting functions for spiking-network simulation results.

This module contains presentation code only.  It does not construct or modify
the Brian2 network, synapses, stimulation, signal filtering, or event detection.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_extended_syn_depression(
        filename,
        output_folder,
        t_array,
        rate_PC,
        rate_PVBC,
        rate_OLM,
        filtered_signal,
        membrane_potential,
        spike_P_t,
        spike_P_i,
        spike_B_t,
        spike_B_i,
        spike_A_t,
        spike_A_i,
        n_raster_pc,
        n_raster_pvbc,
        n_raster_olm,
        output_name=None,
        dpi=600):
    """Create and save the six-row PC/PVBC/OLM, SW, and ripple figure.

    Every argument is a result already produced by
    ``simulate_extended_syn_depression``.  The function performs no simulation
    and does not alter the supplied arrays.

    Returns
    -------
    tuple of str
        Paths of the PNG, PDF, and SVG files.

    Notes
    -----
    ``output_name`` is a filename stem, not a full path.  For example,
    ``output_name='OLM_OLM_0.6_5_seed3'`` creates matching PNG, PDF, and SVG
    files without changing the simulation code.
    """
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['savefig.dpi'] = dpi
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    color_pc = '#3498DB'
    color_pvbc = '#F39C12'
    color_olm = '#1ABC9C'
    color_sharp = '#2E86DE'
    color_ripple = '#F3A6D4'

    t_plot = np.asarray(t_array, dtype=float)
    pc_plot = np.asarray(rate_PC, dtype=float)
    pvbc_plot = np.asarray(rate_PVBC, dtype=float)
    olm_plot = np.asarray(rate_OLM, dtype=float)
    sharp_plot = np.asarray(filtered_signal, dtype=float)
    ripple_plot = np.asarray(membrane_potential, dtype=float)

    fig, axes = plt.subplots(
        6, 1,
        figsize=(8, 7.8),
        dpi=300,
        sharex=True,
        gridspec_kw={
            'height_ratios': [1.55, 0.82, 0.82, 0.82, 1.0, 1.0],
            'hspace': 0.08
        }
    )
    fig.patch.set_facecolor('white')

    # Row 1: sampled-cell raster.  Keep PC at the top, PVBC in the middle,
    # and OLM at the bottom, matching the existing publication figure.
    ax = axes[0]
    gap = 14
    olm_offset = 0
    pvbc_offset = n_raster_olm + gap
    pc_offset = n_raster_olm + gap + n_raster_pvbc + gap

    ax.scatter(
        spike_P_t, np.asarray(spike_P_i, dtype=float) + pc_offset,
        s=8, c=color_pc, marker='.', linewidths=0, rasterized=True
    )
    ax.scatter(
        spike_B_t, np.asarray(spike_B_i, dtype=float) + pvbc_offset,
        s=8, c=color_pvbc, marker='.', linewidths=0, rasterized=True
    )
    ax.scatter(
        spike_A_t, np.asarray(spike_A_i, dtype=float) + olm_offset,
        s=8, c=color_olm, marker='.', linewidths=0, rasterized=True
    )

    olm_mid = olm_offset + (n_raster_olm - 1) / 2
    pvbc_mid = pvbc_offset + (n_raster_pvbc - 1) / 2
    pc_mid = pc_offset + (n_raster_pc - 1) / 2
    ax.set_yticks([olm_mid, pvbc_mid, pc_mid])
    ax.set_yticklabels(['OLM', 'PVBC', 'PC'])

    # Rows 2-4: population firing rates.
    axes[1].plot(t_plot, pc_plot, color=color_pc, lw=1.6)
    axes[1].set_ylim(bottom=0)
    axes[2].plot(t_plot, pvbc_plot, color=color_pvbc, lw=1.6)
    axes[2].set_ylim(bottom=0)
    axes[3].plot(t_plot, olm_plot, color=color_olm, lw=1.6)
    axes[3].set_ylim(bottom=0)

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.08,
        top=0.84,
        hspace=0.08
    )
    pos_pc = axes[1].get_position()
    pos_olm = axes[3].get_position()
    firing_rate_y_center = (pos_pc.y1 + pos_olm.y0) / 2.0
    fig.text(
        0.12,
        firing_rate_y_center,
        'Firing rate (spikes/s)',
        rotation=90,
        ha='center',
        va='center',
        fontsize=14,
        fontweight='bold'
    )

    # Rows 5-6: sharp-wave and ripple-band signals produced by the original
    # filtering code in simulate_spiking.py.
    axes[4].plot(t_plot, sharp_plot, color=color_sharp, lw=1.4)
    axes[4].axhline(0, color='0.75', lw=0.7, zorder=0)
    axes[4].set_ylabel('SW', fontweight='bold')

    axes[5].plot(t_plot, ripple_plot, color=color_ripple, lw=1.0)
    axes[5].axhline(0, color='0.80', lw=0.7, zorder=0)
    axes[5].set_ylabel('Ripple', fontweight='bold')
    axes[5].set_xlabel('Time (ms)', fontweight='bold')

    for ax in axes:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.tick_params(
            axis='both', which='major', direction='out',
            width=1.0, length=4, pad=3
        )
        for ticklab in ax.get_yticklabels():
            ticklab.set_fontweight('bold')
        ax.margins(x=0)

    for ax in axes[:-1]:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axes[-1].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    for ticklab in axes[-1].get_xticklabels():
        ticklab.set_fontweight('bold')

    if len(t_plot) > 1:
        for ax in axes:
            ax.set_xlim(t_plot[0], t_plot[-1])
    if len(pc_plot) > 0:
        axes[1].set_ylim(0, max(10, np.ceil(np.max(pc_plot) / 10.0) * 10 + 5))
    if len(pvbc_plot) > 0:
        axes[2].set_ylim(0, max(10, np.ceil(np.max(pvbc_plot) / 10.0) * 10 + 5))
    if len(olm_plot) > 0:
        axes[3].set_ylim(0, max(5, np.ceil(np.max(olm_plot) / 2.0) * 2 + 1))

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    save_dir = os.path.join(output_folder, 'figures')
    os.makedirs(save_dir, exist_ok=True)

    if output_name is None:
        output_name = filename + 'OLMOLM_0.6,5'
    output_name = os.path.basename(str(output_name))
    name_stem, name_extension = os.path.splitext(output_name)
    if name_extension.lower() in {'.png', '.pdf', '.svg'}:
        output_name = name_stem
    if not output_name:
        raise ValueError('output_name must not be empty')

    output_stem = os.path.join(save_dir, output_name)
    png_path = output_stem + '.png'
    pdf_path = output_stem + '.pdf'
    svg_path = output_stem + '.svg'

    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print('Figure saved:')
    print(' PNG:', png_path)
    print(' PDF:', pdf_path)
    print(' SVG:', svg_path)
    return png_path, pdf_path, svg_path
