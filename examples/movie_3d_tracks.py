"""
3D Track Visualization Movie
============================

This script creates a 3D animation of particle tracking results,
showing particle positions and their trajectories over time.

Prerequisites:
    1. Run demo_synthetic_data.py first to generate test data
       (or synthetic data will be generated automatically)
    2. ffmpeg for MP4 output (optional, will fallback to GIF)

Usage:
    python movie_3d_tracks.py [--output movie.mp4] [--fps 10] [--rotate]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def create_3d_track_movie(
    tracks: "pd.DataFrame",
    voxel_size: tuple[float, float, float],
    output_path: Path,
    fps: int = 10,
    rotate: bool = False,
    trail_length: int = 5,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Create a 3D animation of particle tracks.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks DataFrame with columns [particle, frame, z, y, x]
    voxel_size : tuple[float, float, float]
        Physical voxel size (z_um, y_um, x_um)
    output_path : Path
        Output file path (.mp4 or .gif)
    fps : int
        Frames per second
    rotate : bool
        Whether to rotate the view during animation
    trail_length : int
        Number of past frames to show as trail
    figsize : tuple[int, int]
        Figure size in inches
    """
    import pandas as pd

    z_um, y_um, x_um = voxel_size

    # Convert to physical coordinates
    tracks_phys = tracks.copy()
    tracks_phys["z_um"] = tracks["z"] * z_um
    tracks_phys["y_um"] = tracks["y"] * y_um
    tracks_phys["x_um"] = tracks["x"] * x_um

    # Get frame range and particle IDs
    frames = sorted(tracks_phys["frame"].unique())
    particles = sorted(tracks_phys["particle"].unique())
    n_particles = len(particles)

    # Color map for particles
    colors = plt.cm.turbo(np.linspace(0, 1, n_particles))
    particle_colors = {p: colors[i] for i, p in enumerate(particles)}

    # Calculate axis limits
    x_min, x_max = tracks_phys["x_um"].min(), tracks_phys["x_um"].max()
    y_min, y_max = tracks_phys["y_um"].min(), tracks_phys["y_um"].max()
    z_min, z_max = tracks_phys["z_um"].min(), tracks_phys["z_um"].max()

    # Add padding
    padding = 2.0
    x_min, x_max = x_min - padding, x_max + padding
    y_min, y_max = y_min - padding, y_max + padding
    z_min, z_max = z_min - padding, z_max + padding

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        return []

    def update(frame_idx):
        ax.cla()
        current_frame = frames[frame_idx]

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        ax.set_title(f"Frame {current_frame} / {frames[-1]}")

        # Rotate view if requested
        if rotate:
            ax.view_init(elev=20, azim=frame_idx * 2)

        # Draw each particle
        for particle_id in particles:
            particle_data = tracks_phys[tracks_phys["particle"] == particle_id]
            color = particle_colors[particle_id]

            # Get trail data (past frames)
            trail_start = max(0, frame_idx - trail_length)
            trail_frames = frames[trail_start : frame_idx + 1]
            trail_data = particle_data[particle_data["frame"].isin(trail_frames)]

            if len(trail_data) == 0:
                continue

            # Draw trail as line
            if len(trail_data) > 1:
                ax.plot(
                    trail_data["x_um"],
                    trail_data["y_um"],
                    trail_data["z_um"],
                    color=color,
                    alpha=0.5,
                    linewidth=1.5,
                )

            # Draw current position
            current_pos = particle_data[particle_data["frame"] == current_frame]
            if len(current_pos) > 0:
                ax.scatter(
                    current_pos["x_um"],
                    current_pos["y_um"],
                    current_pos["z_um"],
                    color=color,
                    s=100,
                    edgecolors="black",
                    linewidths=0.5,
                )

        # Draw full trajectories as faint lines
        for particle_id in particles:
            particle_data = tracks_phys[tracks_phys["particle"] == particle_id]
            past_data = particle_data[particle_data["frame"] <= current_frame]
            if len(past_data) > 1:
                ax.plot(
                    past_data["x_um"],
                    past_data["y_um"],
                    past_data["z_um"],
                    color=particle_colors[particle_id],
                    alpha=0.2,
                    linewidth=0.5,
                )

        return []

    print(f"Creating animation with {len(frames)} frames...")
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(frames), interval=1000 // fps, blit=False)

    # Save animation
    output_path = Path(output_path)
    print(f"Saving to {output_path}...")

    if output_path.suffix.lower() == ".mp4":
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(str(output_path), writer=writer)
        except Exception as e:
            print(f"FFmpeg not available ({e}), saving as GIF instead...")
            output_path = output_path.with_suffix(".gif")
            anim.save(str(output_path), writer="pillow", fps=fps)
    else:
        anim.save(str(output_path), writer="pillow", fps=fps)

    plt.close(fig)
    print(f"Saved animation to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 3D track visualization movie")
    parser.add_argument(
        "--output",
        type=str,
        default="./output/track_movie_3d.mp4",
        help="Output file path (.mp4 or .gif)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--rotate", action="store_true", help="Rotate view during animation")
    parser.add_argument("--trail", type=int, default=5, help="Number of frames to show as trail")
    args = parser.parse_args()

    from pt3d.config import DetectionConfig, TrackingConfig, VoxelSize
    from pt3d.detect import detect_batch
    from pt3d.postprocess import filter_stubs
    from pt3d.track import link_detections

    # Load or generate data
    data_path = Path("./output/synthetic/moving_particles_4d.npy")
    if data_path.exists():
        volumes = np.load(data_path)
        print(f"Loaded data from: {data_path}")
    else:
        print("Generating synthetic 4D data...")
        from pt3d.synth import generate_moving_particles

        volumes, _ = generate_moving_particles(
            shape=(20, 32, 64, 64),
            n_particles=10,
            particle_sigma=(2.0, 3.0, 3.0),
            velocity_range=(0.5, 2.0),
            seed=42,
        )

    print(f"Data shape: {volumes.shape} (T, Z, Y, X)")

    # Detection & Tracking
    voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)

    print("\nRunning detection...")
    detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=0.1)
    detections = detect_batch(volumes, detect_config)
    print(f"Detected {len(detections)} particles")

    print("\nRunning tracking...")
    track_config = TrackingConfig(search_range_um=2.0, memory=2)
    tracks = link_detections(detections, track_config, voxel_size)
    tracks = filter_stubs(tracks, min_length=5)
    print(f"Found {tracks['particle'].nunique()} tracks")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create movie
    create_3d_track_movie(
        tracks=tracks,
        voxel_size=voxel_size.as_tuple(),
        output_path=output_path,
        fps=args.fps,
        rotate=args.rotate,
        trail_length=args.trail,
    )


if __name__ == "__main__":
    main()
