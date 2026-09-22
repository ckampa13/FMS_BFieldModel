'''
Visualise the multipole assembly: conductors in 3D, a transverse cross-section,
the z layout, and thin-wire field maps.

Handles both closures.  Arc elements are drawn as the curves they are (via
helicalc.multipole.as_polylines), not as straight chords between their endpoints.

    python plot_multipole_geometry.py -g Multipole_HLLHC_V1_arc
    python plot_multipole_geometry.py -g Multipole_HLLHC_V1_arc -e MQ
    python plot_multipole_geometry.py --no-field          # geometry only, fast
'''
import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial.transform import Rotation

from helicalc.multipole import (MultipoleGeom, load_assembly_csv, as_polylines,
                                element_endpoints, ASSEMBLY_ELEMENTS,
                                MAPPING_VOLUME, min_conductor_radius,
                                thin_wire_field, harmonics_from_field)

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860',
           '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD', '#4878CF', '#EE854A',
           '#6ACC64']


def element_colors(df):
    els = list(dict.fromkeys(df['element']))
    return {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(els)}


def is_axial(row, p0, p1):
    '''True for a run essentially parallel to the beam axis.'''
    d = p1 - p0
    L = np.linalg.norm(d)
    return L > 0 and abs(d[2]) / L > 0.99


# ---------------------------------------------------------------- 3D (plotly)
def plot_3d(df, geom, path, title, n_seg=20, show_cylinder=True):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('  plotly not available, skipping the 3D view')
        return None
    colors = element_colors(df)
    polys = as_polylines(df, n_seg=n_seg)
    fig = go.Figure()
    for el in dict.fromkeys(df['element']):
        idx = np.where((df['element'] == el).values)[0]
        xs, ys, zs = [], [], []
        for k in idx:
            pts = polys[k][0]
            xs += list(pts[:, 0]) + [None]
            ys += list(pts[:, 1]) + [None]
            zs += list(pts[:, 2]) + [None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', name=el,
                                   line=dict(width=3, color=colors[el]),
                                   hovertemplate=el + '<extra></extra>'))
    if show_cylinder:
        th = np.linspace(0, 2 * np.pi, 80)
        zc = np.array([MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1']])
        TH, ZC = np.meshgrid(th, zc)
        fig.add_trace(go.Surface(x=geom.R_map * np.cos(TH),
                                 y=geom.R_map * np.sin(TH), z=ZC,
                                 showscale=False, opacity=0.2,
                                 colorscale=[[0, '#888'], [1, '#888']],
                                 name='mapping cylinder', showlegend=True,
                                 hoverinfo='skip'))
    # Beam axis is the scene's z, which plotly draws VERTICALLY by default.
    # up = scene y puts the transverse vertical axis up the screen, and an eye
    # placed essentially along scene x leaves the beam axis running left-right.
    camera = dict(up=dict(x=0, y=1, z=0),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=1.9, y=0.55, z=0.12))
    fig.update_layout(title=title,
                      scene=dict(xaxis_title='x [m]  (transverse)',
                                 yaxis_title='y [m]  (up)',
                                 zaxis_title='z [m]  (beam axis)',
                                 aspectmode='data', camera=camera),
                      legend=dict(itemsizing='constant'))
    fig.write_html(path)
    png = None
    try:
        png = path.replace('.html', '.png')
        fig.write_image(png, width=1500, height=900, scale=1)
    except Exception as exc:
        print('  (static export skipped: %s)' % str(exc)[:60])
        png = None
    return path, png


def _fill_3d_axes(ax, aspect, dist=8.2, rect=(0.03, 0.02, 0.94, 0.96)):
    """Make a 3D axes actually fill its figure.

    mplot3d inscribes the aspect-scaled box in a square and leaves a wide margin,
    so a long thin magnet ends up as a small strip in a mostly empty frame.
    set_position plus a reduced camera distance (ax.dist, default 10) fixes it.
    """
    try:
        ax.set_box_aspect((aspect, 1, 1))
    except Exception:
        pass
    ax.set_position(list(rect))
    try:
        ax.dist = dist
    except Exception:
        pass


def plot_3d_views(df, geom, path, n_seg=16, views=((18, -62), (18, -110),
                                                    (34, -78), (6, -90))):
    """Several fixed viewpoints in one PNG -- no WebGL, no browser required.

    plotly's 3D needs WebGL, which is unavailable on plenty of setups (remote
    sessions, hardware acceleration off, old drivers).  This and the rotating
    animation below are the fallbacks that always work.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    colors = element_colors(df)
    polys = as_polylines(df, n_seg=n_seg)
    allpts = np.vstack([p[0] for p in polys])
    z0, z1 = allpts[:, 2].min(), allpts[:, 2].max()
    rmax = np.hypot(allpts[:, 0], allpts[:, 1]).max()
    pad = 0.04 * max(z1 - z0, 1e-3)
    z0, z1 = z0 - pad, z1 + pad
    # a long thin magnet stacks better than it tiles: one wide panel per row
    fig = plt.figure(figsize=(11, 3.1 * len(views)))
    for i, (e, a) in enumerate(views):
        ax = fig.add_subplot(len(views), 1, i + 1, projection='3d')
        for el in dict.fromkeys(df['element']):
            idx = np.where((df['element'] == el).values)[0]
            first = True
            for k in idx:
                pts = polys[k][0]
                ax.plot(pts[:, 2], pts[:, 0], pts[:, 1], lw=0.5,
                        color=colors[el], label=el if first else None)
                first = False
        th = np.linspace(0, 2 * np.pi, 60)
        for z in (max(z0, MAPPING_VOLUME['z0']), min(z1, MAPPING_VOLUME['z1'])):
            ax.plot(np.full_like(th, z), geom.R_map * np.cos(th),
                    geom.R_map * np.sin(th), color='k', lw=0.8, alpha=0.6)
        ax.set_xlim(z0, z1); ax.set_ylim(-rmax, rmax); ax.set_zlim(-rmax, rmax)
        ax.set_xlabel('z [m]', fontsize=7, labelpad=1)
        ax.set_ylabel('x [m]', fontsize=7, labelpad=1)
        ax.set_zlabel('y [m]', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=5, pad=0)
        ax.view_init(elev=e, azim=a)
        ax.set_title('elev=%d, azim=%d' % (e, a), fontsize=8)
        _fill_3d_axes(ax, min(max(z1 - z0, 1e-6) / (2 * rmax), 8.0),
                      rect=(0.03, 0.02 + (len(views) - 1 - i) / len(views),
                            0.94, 1.0 / len(views) - 0.04))
        if i == 0:
            ax.legend(ncol=7, fontsize=5, loc='upper center',
                      bbox_to_anchor=(0.5, 1.28))
    fig.suptitle('Multipole assembly conductors -- %d viewpoints' % len(views),
                 fontsize=12, y=0.995)
    fig.subplots_adjust(top=0.90, bottom=0.03, hspace=0.0)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_3d_animation(df, geom, path, n_seg=12, n_frames=72, fps=12):
    """Rotating 3D animation (GIF or MP4 by extension) -- the WebGL-free way to
    actually see the geometry in three dimensions."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    colors = element_colors(df)
    polys = as_polylines(df, n_seg=n_seg)
    allpts = np.vstack([p[0] for p in polys])
    z0, z1 = allpts[:, 2].min(), allpts[:, 2].max()
    rmax = np.hypot(allpts[:, 0], allpts[:, 1]).max()
    pad = 0.04 * max(z1 - z0, 1e-3)
    z0, z1 = z0 - pad, z1 + pad
    # mplot3d shrinks the drawing inside its axes when box_aspect is extreme, so
    # shape the FIGURE to the geometry instead of leaving a mostly-empty frame
    aspect = min(max(z1 - z0, 1e-6) / (2 * rmax), 8.0)
    fig = plt.figure(figsize=(12, max(3.0, 12.0 / max(aspect, 1.2) * 0.9)))
    ax = fig.add_subplot(111, projection='3d')
    for el in dict.fromkeys(df['element']):
        idx = np.where((df['element'] == el).values)[0]
        first = True
        for k in idx:
            pts = polys[k][0]
            ax.plot(pts[:, 2], pts[:, 0], pts[:, 1], lw=0.5, color=colors[el],
                    label=el if first else None)
            first = False
    th = np.linspace(0, 2 * np.pi, 60)
    for z in (max(z0, MAPPING_VOLUME['z0']), min(z1, MAPPING_VOLUME['z1'])):
        ax.plot(np.full_like(th, z), geom.R_map * np.cos(th),
                geom.R_map * np.sin(th), color='k', lw=0.8, alpha=0.6)
    ax.set_xlim(z0, z1); ax.set_ylim(-rmax, rmax); ax.set_zlim(-rmax, rmax)
    # axes off: at ~21:1 the frame and tick labels crowd out the magnet itself.
    # Legend and scale go in the figure margin instead.
    ax.set_axis_off()
    _fill_3d_axes(ax, aspect, dist=7.6, rect=(0.0, 0.0, 1.0, 0.88))
    handles = [plt.Line2D([], [], color=colors[e], lw=2)
               for e in dict.fromkeys(df['element'])]
    fig.legend(handles, list(dict.fromkeys(df['element'])), ncol=7,
               fontsize=6.5, loc='upper center', frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.012,
             'z = %.2f to %.2f m;  black rings = mapping volume, r = %.0f mm'
             % (z0, z1, 1e3 * geom.R_map), ha='center', fontsize=7, color='0.3')

    def frame(i):
        ax.view_init(elev=18 + 10 * np.sin(2 * np.pi * i / n_frames),
                     azim=-180 + 360 * i / n_frames)
        return ()

    anim = FuncAnimation(fig, frame, frames=n_frames, blit=False)
    if path.endswith('.mp4') and FFMpegWriter.isAvailable():
        anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=110)
    else:
        path = path.rsplit('.', 1)[0] + '.gif'
        anim.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return path


# ------------------------------------------------------- transverse section
def plot_cross_section(df, geom, path, element=None):
    sub = df if element is None else df[df.element == element]
    p0, p1 = element_endpoints(sub)
    colors = element_colors(df)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    polys = as_polylines(sub, n_seg=16)
    rmax = geom.a
    for k, (_, row) in enumerate(sub.iterrows()):
        el = row['element']
        if is_axial(row, p0[k], p1[k]):
            # body (or return) bar seen end-on: draw the real W x T rectangle
            R = Rotation.from_euler('zyz', np.array([row['Phi2'], row['theta2'],
                                                     row['psi2']])[::-1],
                                    degrees=True)
            du = R.apply([1, 0, 0])[:2] * row['W'] / 2
            dv = R.apply([0, 1, 0])[:2] * row['T'] / 2
            c = p0[k][:2]
            corners = np.array([c - du - dv, c + du - dv, c + du + dv,
                                c - du + dv, c - du - dv])
            ax.plot(corners[:, 0], corners[:, 1], lw=0.8, color=colors[el])
            # current out of / into the page.  For a racetrack every element of
            # a loop carries the same signed I and the DIRECTION lives in the
            # Euler angles, so the flow sense is sign(I) * sign(dz), not sign(I).
            flow_z = np.sign(row['I']) * np.sign((p1[k] - p0[k])[2])
            ax.plot(*c, marker='+' if flow_z > 0 else '_', ms=4,
                    color=colors[el])
            rmax = max(rmax, np.hypot(*c) + row['T'])
        else:
            pts = polys[k][0]
            ax.plot(pts[:, 0], pts[:, 1], lw=0.4, alpha=0.3, color=colors[el])
            rmax = max(rmax, np.hypot(pts[:, 0], pts[:, 1]).max())
    for r, lab, style in ((geom.R_map, 'mapping r=%.0f mm' % (1e3 * geom.R_map), '-'),
                          (geom.R_aper, 'aperture r=%.0f mm' % (1e3 * geom.R_aper), '--'),
                          (geom.R_ref, 'R_ref=%.0f mm' % (1e3 * geom.R_ref), ':')):
        ax.add_patch(Circle((0, 0), r, fill=False, ls=style, color='k', lw=1.2))
        ax.plot([], [], ls=style, color='k', label=lab)
    rmin = min_conductor_radius(sub)
    ax.add_patch(Circle((0, 0), rmin, fill=False, color='r', lw=1.0, alpha=0.7))
    ax.plot([], [], color='r', alpha=0.7,
            label='min conductor r=%.0f mm' % (1e3 * rmin))
    lim = 1.1 * rmax
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='x [m]', ylabel='y [m]',
           title='Transverse cross-section%s\n(+ / - = current out of / into page)'
                 % ('' if element is None else ' -- ' + element))
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_cross_section_grid(df, geom, path, ncol=5):
    """One small transverse section per element.

    Superimposing all thirteen windings in a single panel is accurate but
    unreadable, and the per-element pattern (which pole order, which sectors
    carry current, where the returns sit) is the thing worth seeing.
    """
    els = [e for e in dict.fromkeys(df['element'])]
    colors = element_colors(df)
    spec_by_name = {s['name']: s for s in ASSEMBLY_ELEMENTS}
    nrow = int(np.ceil(len(els) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, el in zip(axes, els):
        sub = df[df.element == el]
        p0, p1 = element_endpoints(sub)
        polys = as_polylines(sub, n_seg=12)
        rmax = geom.a
        for k, (_, row) in enumerate(sub.iterrows()):
            if is_axial(row, p0[k], p1[k]):
                c = p0[k][:2]
                flow = np.sign(row['I']) * np.sign((p1[k] - p0[k])[2])
                ax.plot(*c, marker='+' if flow > 0 else '_', ms=5,
                        color='#C44E52' if flow > 0 else '#4C72B0', mew=1.2)
                rmax = max(rmax, np.hypot(*c))
            else:
                pts = polys[k][0]
                ax.plot(pts[:, 0], pts[:, 1], lw=0.35, alpha=0.25,
                        color=colors[el])
                rmax = max(rmax, np.hypot(pts[:, 0], pts[:, 1]).max())
        for r, st in ((geom.R_map, '-'), (geom.R_ref, ':')):
            ax.add_patch(Circle((0, 0), r, fill=False, ls=st, color='k', lw=0.9))
        sp = spec_by_name.get(el)
        ttl = el if sp is None else '%s  2n=%d%s' % (el, 2 * sp['n'],
                                                     ' skew' if sp['skew'] else '')
        ax.set_title(ttl, fontsize=9)
        lim = 1.12 * rmax
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[])
        ax.set_aspect('equal')
    for ax in axes[len(els):]:
        ax.axis('off')
    fig.suptitle('Transverse sections, one per winding\n'
                 'red + = current out of page, blue - = into page; '
                 'circles are R_map and R_ref', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- z layout
def plot_layout(df, geom, path):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    colors = element_colors(df)
    for i, spec in enumerate(ASSEMBLY_ELEMENTS):
        y = 0.35 if i % 2 else 0.95
        ax.plot([spec['z0'], spec['z1']], [y, y], lw=10, solid_capstyle='butt',
                color=colors.get(spec['name'], '#888'))
        ax.text(0.5 * (spec['z0'] + spec['z1']), y + 0.09,
                '%s\n2n=%d%s' % (spec['name'], 2 * spec['n'],
                                 ' skew' if spec['skew'] else ''),
                ha='center', va='bottom', fontsize=7)
    if 'PERT' in set(df['element']):
        pz = df[df.element == 'PERT']['z0'].mean()
        ax.plot([pz], [0.15], marker='*', ms=14, color='k')
        ax.text(pz, 0.02, 'PERT', ha='center', va='bottom', fontsize=7)
    ax.axvspan(MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1'], color='0.9', zorder=0)
    ax.set(xlabel='z [m]', ylim=(0, 1.55), yticks=[],
           title='Assembly layout (grey = mapping volume)')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------- field maps
def plot_field_maps(df, geom, path, n_grid=61, n_z=260):
    '''Thin-wire field: transverse maps at four element centres + axial profile.

    Thin wire (no GPU) is plenty for a picture -- it tracks the full 3D field to
    well under a percent.
    '''
    picks = ['MQ', 'MCBH', 'MCS', 'MCT']
    specs = [s for s in ASSEMBLY_ELEMENTS if s['name'] in picks]
    R = geom.R_map
    g = np.linspace(-R, R, n_grid)
    X, Y = np.meshgrid(g, g)
    inside = np.hypot(X, Y) <= R

    fig = plt.figure(figsize=(15, 8.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85], hspace=0.30, wspace=0.55)
    for j, spec in enumerate(specs):
        zc = 0.5 * (spec['z0'] + spec['z1'])
        P = np.column_stack([X[inside], Y[inside], np.full(inside.sum(), zc)])
        B = thin_wire_field(df[df.element == spec['name']], P)
        mag = np.full(X.shape, np.nan)
        mag[inside] = 1e4 * np.linalg.norm(B[:, :2], axis=1)
        ax = fig.add_subplot(gs[0, j])
        im = ax.pcolormesh(X, Y, mag, shading='auto', cmap='viridis')
        # transverse field direction
        s = max(n_grid // 14, 1)
        Us = np.full(X.shape, np.nan); Vs = np.full(X.shape, np.nan)
        Us[inside] = B[:, 0]; Vs[inside] = B[:, 1]
        ax.quiver(X[::s, ::s], Y[::s, ::s], Us[::s, ::s], Vs[::s, ::s],
                  color='w', scale_units='xy', angles='xy', width=0.006,
                  alpha=0.85)
        ax.add_patch(Circle((0, 0), geom.R_ref, fill=False, ls=':', color='w', lw=1))
        ax.set_aspect('equal')
        ax.set_title('%s  (2n=%d%s)\nz = %.2f m' %
                     (spec['name'], 2 * spec['n'],
                      ' skew' if spec['skew'] else '', zc), fontsize=9)
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('y [m]', fontsize=8, labelpad=1)
        ax.tick_params(labelsize=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cb.set_label('|B_transverse| [G]', fontsize=7)
        cb.ax.tick_params(labelsize=6)
        cb.ax.yaxis.get_offset_text().set_fontsize(6)

    # axial profile: each element's own main harmonic along z
    ax = fig.add_subplot(gs[1, :])
    M = 32
    ph = 2 * np.pi * np.arange(M) / M
    zs = np.linspace(MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1'], n_z)
    colors = element_colors(df)
    for spec in ASSEMBLY_ELEMENTS:
        sub = df[df.element == spec['name']]
        # one call for the whole profile: thin_wire_field loops over segments in
        # python, so calling it per z would be ~260x slower for no reason
        pts = np.vstack([np.column_stack([geom.R_ref * np.cos(ph),
                                          geom.R_ref * np.sin(ph),
                                          np.full(M, z)]) for z in zs])
        Ball = thin_wire_field(sub, pts)
        prof = []
        for i in range(len(zs)):
            B = Ball[i * M:(i + 1) * M]
            h = harmonics_from_field(B[:, 0], B[:, 1], ph, n_max=8)
            prof.append(h[spec['n']][1] if spec['skew'] else h[spec['n']][0])
        ax.plot(zs, np.array(prof), lw=1.4, color=colors[spec['name']],
                label='%s (2n=%d%s)' % (spec['name'], 2 * spec['n'],
                                        ' skew' if spec['skew'] else ''))
    ax.axhline(0, color='k', lw=0.5)
    ax.set(xlabel='z [m]', ylabel='own main harmonic at R_ref [T]',
           title='Each winding\'s own field along the axis (thin wire, r = %.0f mm)'
                 % (1e3 * geom.R_ref))
    ax.legend(ncol=5, fontsize=7, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_3d_static(df, geom, path, title, n_seg=16, elev=20, azim=-62):
    """Static preview: a 3D perspective plus a side elevation.

    Exists because plotly's static export needs the kaleido binary, which is not
    installed here -- the interactive HTML is still the primary 3D view.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    colors = element_colors(df)
    polys = as_polylines(df, n_seg=n_seg)
    allpts = np.vstack([p[0] for p in polys])
    z0, z1 = allpts[:, 2].min(), allpts[:, 2].max()
    rmax = np.hypot(allpts[:, 0], allpts[:, 1]).max()
    pad = 0.04 * max(z1 - z0, 1e-3)
    z0, z1 = z0 - pad, z1 + pad

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)
    ax = fig.add_subplot(gs[0], projection='3d')
    for el in dict.fromkeys(df['element']):
        idx = np.where((df['element'] == el).values)[0]
        first = True
        for k in idx:
            pts = polys[k][0]
            ax.plot(pts[:, 2], pts[:, 0], pts[:, 1], lw=0.6, color=colors[el],
                    label=el if first else None)
            first = False
    th = np.linspace(0, 2 * np.pi, 60)
    for z in (max(z0, MAPPING_VOLUME['z0']), min(z1, MAPPING_VOLUME['z1'])):
        ax.plot(np.full_like(th, z), geom.R_map * np.cos(th),
                geom.R_map * np.sin(th), color='k', lw=0.9, alpha=0.6)
    ax.set_xlim(z0, z1); ax.set_ylim(-rmax, rmax); ax.set_zlim(-rmax, rmax)
    ax.set_xlabel('z [m]', fontsize=8, labelpad=2)
    ax.set_ylabel('x [m]', fontsize=8, labelpad=2)
    ax.set_zlabel('y [m]', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    try:
        span = max(z1 - z0, 1e-6)
        ax.set_box_aspect((min(span / (2 * rmax), 6.0), 1, 1))
    except Exception:
        pass
    n_el = len(set(df['element']))
    ax.legend(ncol=2 if n_el > 6 else 1, fontsize=6, loc='upper left')

    # RADIUS vs z, not an x-z projection.  A projection is misleading here: a
    # conductor at theta ~ 90 deg has x ~ 0 and would appear to sit inside the
    # mapping volume while actually being out at r = a.  Plotting r(z) shows the
    # clearance honestly.
    ax2 = fig.add_subplot(gs[1])
    for el in dict.fromkeys(df['element']):
        idx = np.where((df['element'] == el).values)[0]
        for k in idx:
            pts = polys[k][0]
            ax2.plot(pts[:, 2], np.hypot(pts[:, 0], pts[:, 1]), lw=0.5,
                     color=colors[el], alpha=0.8)
    ax2.axhspan(0, geom.R_map, color='0.88', zorder=0)
    ax2.axhline(geom.R_map, color='k', lw=1.0)
    ax2.axhline(geom.R_aper, color='k', lw=0.9, ls='--')
    ax2.text(z0 + 0.01 * (z1 - z0), geom.R_map, ' mapping r=%.0f mm'
             % (1e3 * geom.R_map), va='bottom', fontsize=7)
    ax2.text(z0 + 0.01 * (z1 - z0), geom.R_aper, ' aperture r=%.0f mm'
             % (1e3 * geom.R_aper), va='bottom', fontsize=7)
    ax2.set(xlabel='z [m]', ylabel='conductor radius  r [m]', xlim=(z0, z1),
            ylim=(0, None), title='Radial clearance along the beam axis')
    ax2.grid(alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-g', '--Geom', default='Multipole_HLLHC_V1_arc')
    p.add_argument('-e', '--element', default=None,
                   help='restrict the 3D and cross-section views to one element')
    p.add_argument('--aperture', type=float, default=0.150)
    p.add_argument('--winding-radius', default='0.090')
    p.add_argument('--out-dir', default=None,
                   help='default: <helicalc_data>/Bmaps/multipole/plots/')
    p.add_argument('--anim', action='store_true',
                   help='also write a rotating 3D animation (no WebGL needed)')
    p.add_argument('--no-field', action='store_true',
                   help='skip the field maps (they cost a minute of CPU)')
    args = p.parse_args(argv)
    a = None if str(args.winding_radius).lower() in ('auto', 'none', '') \
        else float(args.winding_radius)
    geom = MultipoleGeom(aperture=args.aperture, a=a)
    df = load_assembly_csv(args.Geom)
    out_dir = args.out_dir
    if out_dir is None:
        from helicalc import helicalc_data
        out_dir = os.path.join(helicalc_data, 'Bmaps', 'multipole', 'plots', '')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, args.Geom)
    sub = df if args.element is None else df[df.element == args.element]
    tag = '' if args.element is None else '_' + args.element

    made = []
    r = plot_3d(sub, geom, base + tag + '_conductors_3d.html',
                'Multipole assembly conductors%s' % ('' if args.element is None
                                                     else ' -- ' + args.element))
    if r:
        made += [x for x in r if x]
    made.append(plot_3d_static(sub, geom, base + tag + '_conductors_3d.png',
                               'Multipole assembly%s'
                               % ('' if args.element is None
                                  else ' -- ' + args.element)))
    made.append(plot_3d_views(sub, geom, base + tag + '_conductors_3d_views.png'))
    if args.anim:
        made.append(plot_3d_animation(sub, geom,
                                      base + tag + '_conductors_3d.mp4'))
    made.append(plot_cross_section(df, geom, base + tag + '_cross_section.png',
                                   args.element))
    if args.element is None:
        made.append(plot_cross_section_grid(df, geom,
                                            base + '_cross_sections_grid.png'))
    made.append(plot_layout(df, geom, base + '_layout.png'))
    if not args.no_field:
        made.append(plot_field_maps(df, geom, base + '_field_maps.png'))
    for f in made:
        print('  wrote %s' % f)


if __name__ == '__main__':
    main()
