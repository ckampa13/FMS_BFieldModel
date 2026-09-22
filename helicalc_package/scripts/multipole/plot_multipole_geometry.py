'''
Visualise the multipole assembly conductors.

Produces
  - an interactive plotly 3D view of every bar (html)
  - a matplotlib transverse cross-section showing the mapping cylinder, the
    aperture wall, the winding radius and the end-turn clearance
  - a matplotlib z-layout strip showing where each element sits

    python plot_multipole_geometry.py
    python plot_multipole_geometry.py --element MQ --out-dir /tmp
'''
import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from helicalc.multipole import (MultipoleGeom, load_assembly_csv, bar_endpoints,
                                ASSEMBLY_ELEMENTS, MAPPING_VOLUME,
                                min_conductor_radius)

# one colour per element, stable across figures
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860',
           '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD', '#4878CF', '#EE854A',
           '#6ACC64']


def element_colors(df):
    els = list(dict.fromkeys(df['element']))
    return {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(els)}


def plot_3d(df, geom, path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('  plotly not available, skipping the 3D view')
        return None
    p0, p1 = bar_endpoints(df)
    colors = element_colors(df)
    fig = go.Figure()
    for el in dict.fromkeys(df['element']):
        m = (df['element'] == el).values
        xs, ys, zs = [], [], []
        for A, B in zip(p0[m], p1[m]):
            xs += [A[0], B[0], None]
            ys += [A[1], B[1], None]
            zs += [A[2], B[2], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', name=el,
                                   line=dict(width=3, color=colors[el])))
    # mapping cylinder
    th = np.linspace(0, 2 * np.pi, 60)
    zc = np.array([MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1']])
    TH, ZC = np.meshgrid(th, zc)
    fig.add_trace(go.Surface(x=geom.R_map * np.cos(TH), y=geom.R_map * np.sin(TH),
                             z=ZC, showscale=False, opacity=0.25,
                             colorscale=[[0, '#888'], [1, '#888']],
                             name='mapping cylinder'))
    fig.update_layout(title='Multipole assembly conductors',
                      scene=dict(xaxis_title='x [m]', yaxis_title='y [m]',
                                 zaxis_title='z [m]', aspectmode='data'))
    fig.write_html(path)
    return path


def plot_cross_section(df, geom, path, element=None):
    sub = df if element is None else df[df.element == element]
    p0, p1 = bar_endpoints(sub)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = element_colors(df)
    # body bars as cross-section rectangles, end turns as light traces
    # Draw each body bar's cross-section from its ACTUAL Euler angles rather
    # than from its azimuth, so the figure is a genuine check that psi2 orients
    # W azimuthally and T radially, not a redrawing of that assumption.
    from scipy.spatial.transform import Rotation
    for (_, row), A, B in zip(sub.iterrows(), p0, p1):
        el, kind, I = row['element'], row['seg_kind'], row['I']
        if kind != 'axial':
            ax.plot([A[0], B[0]], [A[1], B[1]], lw=0.4, alpha=0.35,
                    color=colors[el])
            continue
        R = Rotation.from_euler(
            'zyz', np.array([row['Phi2'], row['theta2'], row['psi2']])[::-1],
            degrees=True)
        du = R.apply([1, 0, 0])[:2] * row['W'] / 2      # local x' -> W
        dv = R.apply([0, 1, 0])[:2] * row['T'] / 2      # local y' -> T
        c = np.array([A[0], A[1]])
        corners = np.array([c - du - dv, c + du - dv, c + du + dv,
                            c - du + dv, c - du - dv])
        ax.plot(corners[:, 0], corners[:, 1], lw=0.8, color=colors[el])
        ax.plot(*c, marker='+' if I > 0 else '_', ms=4, color=colors[el])
    for r, lab, style in ((geom.R_map, 'mapping r=%.0f mm' % (1e3 * geom.R_map), '-'),
                          (geom.R_aper, 'aperture r=%.0f mm' % (1e3 * geom.R_aper), '--'),
                          (geom.R_ref, 'R_ref=%.0f mm' % (1e3 * geom.R_ref), ':')):
        ax.add_patch(Circle((0, 0), r, fill=False, ls=style, color='k', lw=1.2))
        ax.plot([], [], ls=style, color='k', label=lab)
    rmin = min_conductor_radius(sub)
    ax.add_patch(Circle((0, 0), rmin, fill=False, color='r', lw=1.0, alpha=0.7))
    ax.plot([], [], color='r', alpha=0.7,
            label='min conductor r=%.0f mm' % (1e3 * rmin))
    lim = 1.25 * geom.a
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='x [m]', ylabel='y [m]',
           title='Transverse cross-section%s' % ('' if element is None
                                                 else ' -- ' + element))
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_layout(df, geom, path):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    colors = element_colors(df)
    # two rows so neighbouring elements do not overlap; labels always sit above
    # their own bar, which keeps them clear of the axis
    for i, spec in enumerate(ASSEMBLY_ELEMENTS):
        y = 0.35 if i % 2 else 0.95
        ax.plot([spec['z0'], spec['z1']], [y, y], lw=10, solid_capstyle='butt',
                color=colors.get(spec['name'], '#888'))
        ax.text(0.5 * (spec['z0'] + spec['z1']), y + 0.09,
                '%s\n2n=%d%s' % (spec['name'], 2 * spec['n'],
                                 ' skew' if spec['skew'] else ''),
                ha='center', va='bottom', fontsize=7)
    if 'PERT' in set(df['element']):
        pz = df[df.element == 'PERT']['z_mid'].mean()
        ax.plot([pz], [0.15], marker='*', ms=14, color='k')
        ax.text(pz, 0.02, 'PERT', ha='center', va='bottom', fontsize=7)
    ax.axvspan(MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1'], color='0.9', zorder=0)
    ax.set(xlabel='z [m]', ylim=(0, 1.55), yticks=[],
           title='Assembly layout (grey = mapping volume)')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-g', '--Geom', default='Multipole_HLLHC_V1')
    p.add_argument('-e', '--element', default=None)
    p.add_argument('--aperture', type=float, default=0.150)
    p.add_argument('--winding-radius', default='0.090')
    p.add_argument('--out-dir', default='.')
    args = p.parse_args(argv)
    a = None if str(args.winding_radius).lower() in ('auto', 'none', '') \
        else float(args.winding_radius)
    geom = MultipoleGeom(aperture=args.aperture, a=a)
    df = load_assembly_csv(args.Geom)
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.Geom)

    for f in (plot_3d(df if args.element is None else df[df.element == args.element],
                      geom, base + '_conductors_3d.html'),
              plot_cross_section(df, geom, base + '_cross_section.png', args.element),
              plot_layout(df, geom, base + '_layout.png')):
        if f:
            print('  wrote %s' % f)


if __name__ == '__main__':
    main()
