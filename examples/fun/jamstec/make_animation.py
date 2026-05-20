"""
Create backward-in-time animation of ICM wavefield.

Reads GMT xyz wavefield snapshots, overlays on the mesh outline,
and produces an MP4 animation with frames in reverse order so that
waves converge toward the "i-dot" source position.

Usage:
    python make_animation.py [--results-dir ./results] [--forward] [--fps 30]
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def parse_abaqus_mesh(inp_file):
    """Parse ABAQUS .inp for quad element edges (for mesh overlay)."""
    nodes = {}
    elements = []
    section = None

    with open(inp_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("**"):
                continue
            if line.startswith("*NODE"):
                section = "node"
                continue
            elif line.startswith("*ELEMENT"):
                section = "element"
                continue
            elif line.startswith("*"):
                section = None
                continue

            if section == "node":
                parts = line.split(",")
                nid = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                nodes[nid] = (x, y)
            elif section == "element":
                parts = [int(p) for p in line.split(",")]
                elements.append(parts[1:])  # skip element ID

    return nodes, elements


def build_mesh_lines(nodes, elements):
    """Build line segments for mesh edges (for LineCollection)."""
    segments = []
    for conn in elements:
        if len(conn) < 4:
            continue
        pts = [nodes[n] for n in conn[:4]]
        for i in range(4):
            j = (i + 1) % 4
            segments.append([pts[i], pts[j]])
    return segments


def load_xyz(filepath):
    """Load GMT xyz file into structured grid arrays."""
    data = np.loadtxt(filepath)
    x = data[:, 0]
    y = data[:, 1]
    v = data[:, 2]

    # Determine grid dimensions
    xu = np.unique(x)
    yu = np.unique(y)
    nx, ny = len(xu), len(yu)

    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    V = v.reshape(ny, nx)

    return X, Y, V


def find_snapshots(results_dir, field="PS"):
    """Find all GMT xyz snapshots and sort by step number.

    For elastic: DISP_mag_000025.xyz, DISP_x_000025.xyz, DISP_y_000025.xyz
    For acoustic: PS_000025.xyz
    """
    gmt_dir = os.path.join(results_dir, "wavefield", "gmt")
    pattern = os.path.join(gmt_dir, f"{field}_*.xyz")
    files = sorted(glob.glob(pattern))

    if not files:
        # Fallback: try PS (acoustic)
        pattern = os.path.join(gmt_dir, "PS_*.xyz")
        files = sorted(glob.glob(pattern))
        field = "PS"

    snapshots = []
    for f in files:
        match = re.search(rf"{field}_(\d+)\.xyz$", f)
        if match:
            step = int(match.group(1))
            snapshots.append((step, f))

    snapshots.sort(key=lambda x: x[0])
    return snapshots, field


def _normalize_abs(V):
    """Normalize absolute value per-frame to [0, 1]."""
    Va = np.abs(V)
    vmax = np.nanmax(Va)
    if vmax < 1e-30:
        return np.zeros_like(V)
    return Va / vmax


def main():
    parser = argparse.ArgumentParser(
        description="Create ICM wavefield animation (backward in time)")
    parser.add_argument("--results-dir", default="./results",
                        help="Simulation results directory")
    parser.add_argument("--mesh", default="jamstec.inp",
                        help="ABAQUS mesh file for overlay")
    parser.add_argument("--forward", action="store_true",
                        help="Normal time order (default: reversed)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second")
    parser.add_argument("--output", default="jamstec_backward.mp4",
                        help="Output animation file (.mp4 or .gif)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Resolution (DPI)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Color scale max (auto if not set)")
    parser.add_argument("--cmap", default="turbo",
                        help="Colormap name (turbo, jet, plasma, inferno, hot)")
    parser.add_argument("--field", default="PS",
                        help="Field name (PS, DISP_mag, DISP_x, DISP_y)")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Animation start time [s] (sim time where anim begins). "
                             "For backward anim, this is the LATEST sim time.")
    parser.add_argument("--t-end", type=float, default=0.6,
                        help="Animation end time [s] (sim time where anim ends). "
                             "For backward anim, this is the EARLIEST sim time. "
                             "Default: 0.6s")
    parser.add_argument("--dt", type=float, default=None,
                        help="Time step (auto-detect from config if not set)")
    args = parser.parse_args()

    # Load mesh
    print(f"Loading mesh: {args.mesh}")
    nodes, elements = parse_abaqus_mesh(args.mesh)
    mesh_segments = build_mesh_lines(nodes, elements)
    print(f"  {len(nodes)} nodes, {len(elements)} elements")

    # Find snapshots
    snapshots, field_name = find_snapshots(args.results_dir, args.field)
    print(f"  Field: {field_name}")
    if not snapshots:
        print(f"No snapshots found in {args.results_dir}/wavefield/gmt/")
        return
    print(f"Found {len(snapshots)} snapshots "
          f"(steps {snapshots[0][0]} to {snapshots[-1][0]})")

    # Auto-detect dt from config.yaml
    dt = args.dt
    if dt is None:
        try:
            import yaml
            with open("config.yaml") as f:
                cfg = yaml.safe_load(f)
            dt = cfg["simulation"]["dt"]
            print(f"  dt from config.yaml: {dt}")
        except Exception:
            # Estimate from snapshot spacing
            if len(snapshots) >= 2:
                step_interval = snapshots[1][0] - snapshots[0][0]
                dt = args.end_time / snapshots[-1][0] if snapshots[-1][0] > 0 else 1e-4
            else:
                dt = 1e-4
            print(f"  dt estimated: {dt}")

    # Filter by time window
    # For backward anim: t_start is the latest sim time, t_end is the earliest
    # For forward anim: t_start is the earliest, t_end is the latest
    t_start = args.t_start
    t_end = args.t_end

    if not args.forward:
        # Backward: t_start=latest sim time, t_end=earliest sim time (where anim ends)
        if t_start is None:
            t_start = snapshots[-1][0] * dt  # last available
        # t_start > t_end in sim time
        step_lo = int(t_end / dt)
        step_hi = int(t_start / dt)
    else:
        # Forward: t_start=earliest, t_end=latest
        if t_start is None:
            t_start = 0.0
        step_lo = int(t_start / dt)
        step_hi = int(t_end / dt)

    snapshots = [(s, f) for s, f in snapshots if step_lo <= s <= step_hi]
    print(f"  Time window: {min(t_start, t_end):.3f}s — {max(t_start, t_end):.3f}s "
          f"(steps {step_lo}–{step_hi}): {len(snapshots)} snapshots")

    # Reverse for backward animation
    if not args.forward:
        snapshots = snapshots[::-1]
        print("  -> Reversed for backward-in-time animation")

    # Load first frame to get grid and auto-detect vmax
    X, Y, V0 = load_xyz(snapshots[0][1])

    if args.vmax is None:
        sample_indices = np.linspace(0, len(snapshots) - 1,
                                     min(20, len(snapshots)), dtype=int)
        vmax = 0.0
        for idx in sample_indices:
            _, _, Vi = load_xyz(snapshots[idx][1])
            vmax = max(vmax, np.nanmax(np.abs(Vi)))
        args.vmax = vmax * 0.8
        print(f"  Auto vmax: {args.vmax:.3e} (80% of global max {vmax:.3e})")

    # Source position
    src_pos = None
    if os.path.exists("source_position.txt"):
        src_pos = np.loadtxt("source_position.txt")

    # Setup figure — dark theme
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Wavefield as imshow — per-frame normalization
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    is_magnitude = "mag" in field_name
    cmap_obj = plt.get_cmap(args.cmap).copy()
    cmap_obj.set_bad(color="black")  # NaN (holes/letters) = black

    # Initial frame
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax.imshow(_normalize_abs(V0), cmap=cmap_obj, norm=norm, origin="lower",
                   extent=extent, aspect="equal", interpolation="bilinear",
                   zorder=0)

    # Mesh overlay (on top of wavefield)
    lc = LineCollection(mesh_segments, colors="white", linewidths=0.4,
                        alpha=1.0, zorder=1)
    ax.add_collection(lc)

    label = "Normalized displacement"
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.label.set_color("white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("y [m]", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    direction = "Forward" if args.forward else "Backward"
    title_text = ax.set_title("", color="white")

    n_frames = len(snapshots)

    def update(frame_idx):
        step, filepath = snapshots[frame_idx]
        _, _, V = load_xyz(filepath)
        im.set_data(_normalize_abs(V))
        t = step * dt
        title_text.set_text(f"JAMSTEC Acoustic — {direction} — t={t:.3f}s (step {step})")
        pct = (frame_idx + 1) / n_frames * 100
        print(f"\r  Rendering: {frame_idx+1}/{n_frames} ({pct:.0f}%)", end="", flush=True)
        return [im, title_text]

    print(f"\nRendering {len(snapshots)} frames at {args.fps} fps...")
    anim = FuncAnimation(fig, update, frames=len(snapshots),
                         blit=True, interval=1000 // args.fps)

    # Save
    if args.output.endswith(".gif"):
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=5000,
                              extra_args=["-vcodec", "libx264",
                                          "-pix_fmt", "yuv420p"])

    anim.save(args.output, writer=writer, dpi=args.dpi)
    print(f"\n  Saved: {args.output}")
    plt.close()

    # Also save a single frame preview
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 7))
    fig2.set_facecolor("black")
    fig2.patch.set_facecolor("black")
    ax2.set_facecolor("black")
    mid = len(snapshots) // 2
    _, _, Vmid = load_xyz(snapshots[mid][1])
    ax2.imshow(_normalize_abs(Vmid), cmap=cmap_obj, norm=norm, origin="lower",
               extent=extent, aspect="equal", interpolation="bilinear")
    lc2 = LineCollection(mesh_segments, colors="white", linewidths=0.4, alpha=1.0)
    ax2.add_collection(lc2)
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_xlabel("x [m]", color="white")
    ax2.set_ylabel("y [m]", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")
    ax2.set_title(f"JAMSTEC Acoustic — Frame {mid} (Step {snapshots[mid][0]})",
                  color="white")
    fig2.tight_layout()
    preview = args.output.rsplit(".", 1)[0] + "_preview.png"
    fig2.savefig(preview, dpi=args.dpi)
    print(f"Saved preview: {preview}")
    plt.close()


if __name__ == "__main__":
    main()
