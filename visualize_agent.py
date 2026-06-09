"""
visualize_agent.py — Watch the trained agent navigate a real city map
----------------------------------------------------------------------
Renders the SLAM belief map being built in real-time alongside ground
truth, with the agent's trajectory and target region overlaid.

Usage:
    python visualize_agent.py
    python visualize_agent.py --city-map data/real_grid.json --target-x 20 --target-y 20
    python visualize_agent.py --speed 0.05   # slow down (seconds per step)
    python visualize_agent.py --save-gif     # save trajectory as GIF
"""

import argparse
import json
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

from envs.city_env import CityExplorerEnv
from agent.mamba_trainer_fast import FastMambaPPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/mamba_fast_hybrid_slam.pt')
    p.add_argument('--city-map', default='data/real_grid.json')
    p.add_argument('--target-x', type=int, default=None)
    p.add_argument('--target-y', type=int, default=None)
    p.add_argument('--target-radius', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=1200)
    p.add_argument('--speed', type=float, default=0.08,
                   help='Seconds to pause between steps (0=max speed)')
    p.add_argument('--save-gif', action='store_true')
    p.add_argument('--out', type=str, default='evaluation_results/agent_run.gif',
                   help='Output path for GIF')
    p.add_argument('--near-start', action='store_true',
                   help='Force agent to start near target (demo mode)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def slam_to_image(slam_map, W, H):
    """Convert SLAM binary_map to RGB: unknown=grey, free=white, occupied=dark."""
    img = np.ones((H, W, 3), dtype=np.float32) * 0.5  # unknown = grey
    free = slam_map == 0
    occ  = slam_map == 1
    img[free] = [0.95, 0.95, 0.95]   # free = near-white
    img[occ]  = [0.25, 0.25, 0.25]   # occupied = dark
    return img


def truth_to_image(obstacles, W, H):
    """Ground truth: white=free, dark=obstacle."""
    img = np.ones((H, W, 3), dtype=np.float32)
    img[obstacles] = [0.25, 0.25, 0.25]
    return img


def overlay_region(img, region_mask, alpha=0.25):
    """Tint target region blue."""
    out = img.copy()
    out[region_mask] = out[region_mask] * (1 - alpha) + np.array([0.2, 0.5, 1.0]) * alpha
    return out


def overlay_trajectory(img, trajectory, W, H, color=(1.0, 0.3, 0.0)):
    """Draw agent trajectory as orange dots (fading)."""
    out = img.copy()
    n = len(trajectory)
    for i, (tx, ty) in enumerate(trajectory):
        if 0 <= tx < W and 0 <= ty < H:
            fade = 0.3 + 0.7 * (i / max(n - 1, 1))
            out[ty, tx] = np.array(color) * fade + out[ty, tx] * (1 - fade)
    return out


def run_visualization(args):
    with open(args.city_map) as f:
        city_map = json.load(f)

    W, H = city_map['W'], city_map['H']

    # Pick target: CLI override > map default > centre
    if args.target_x is not None and args.target_y is not None:
        target = [args.target_x, args.target_y]
    elif 'target' in city_map:
        target = city_map['target']
    else:
        # Pick a free cell near the centre
        obstacles = np.array(
            [item for row in city_map['obstacles'] for item in row], dtype=bool
        ).reshape(H, W)
        cy, cx = H // 2, W // 2
        for r in range(1, max(W, H)):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and not obstacles[ny, nx]:
                        target = [nx, ny]
                        break
                else:
                    continue
                break
            else:
                continue
            break

    print(f"Map:    {city_map.get('name', args.city_map)}  ({W}×{H}, {city_map.get('cell_size',10)}m/cell)")
    print(f"Target: {target}  radius={args.target_radius}")

    env = CityExplorerEnv(
        width=W, height=H,
        max_steps=args.max_steps,
        target_radius=args.target_radius,
        target_coverage=0.85,
        target_score=0.80,
        seed=args.seed,
    )

    trainer = FastMambaPPOTrainer(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        policy_type='fast_hybrid',
        d_model=128,
        n_layers=2,
        memory_size=500,
    )
    trainer.load(args.checkpoint)
    trainer.net.eval()
    trainer.reset_episode()

    if args.near_start:
        # Override curriculum: always start within the target region for demo mode
        env._choose_start = lambda: env._random_free_pos() if True else None
        import types
        def _near_start(self):
            tx, ty = self._target
            r = self._target_radius()
            for _ in range(200):
                angle = env.np_random.uniform(0, 2 * 3.14159)
                dist  = env.np_random.uniform(0, r * 1.2)
                import numpy as _np
                sx = int(_np.clip(tx + dist * _np.cos(angle), 0, self.W - 1))
                sy = int(_np.clip(ty + dist * _np.sin(angle), 0, self.H - 1))
                if not self.city.obstacles[sy, sx]:
                    return _np.array([sx, sy], dtype=_np.int32)
            return self._random_free_pos()
        env._choose_start = types.MethodType(_near_start, env)

    obs, _ = env.reset(seed=args.seed, options={'city_map': city_map, 'target': target})
    region_mask = env._region_mask

    # --- matplotlib setup ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title('SLAM Belief Map', color='white', fontsize=11)
    axes[1].set_title('Ground Truth + Trajectory', color='white', fontsize=11)

    slam_img  = slam_to_image(env.slam.binary_map, W, H)
    truth_img = truth_to_image(env.city.obstacles, W, H)

    slam_disp  = axes[0].imshow(overlay_region(slam_img, region_mask),
                                 origin='upper', vmin=0, vmax=1, interpolation='nearest')
    truth_disp = axes[1].imshow(overlay_region(truth_img, region_mask),
                                 origin='upper', vmin=0, vmax=1, interpolation='nearest')

    # Agent marker
    agent_dot_slam,  = axes[0].plot([], [], 'o', color='#ff4444', ms=6, zorder=5)
    agent_dot_truth, = axes[1].plot([], [], 'o', color='#ff4444', ms=6, zorder=5)

    # Target marker
    axes[0].plot(target[0], target[1], '*', color='#ffdd00', ms=10, zorder=6)
    axes[1].plot(target[0], target[1], '*', color='#ffdd00', ms=10, zorder=6)

    stats_text = fig.text(0.5, 0.02,
        'Step 0 | Coverage 0.0% | Score 0.0% | Reward 0.0',
        ha='center', color='#aaaaaa', fontsize=9)

    legend = [
        mpatches.Patch(color=[0.95,0.95,0.95], label='Free'),
        mpatches.Patch(color=[0.25,0.25,0.25], label='Obstacle'),
        mpatches.Patch(color=[0.5,0.5,0.5],    label='Unknown'),
        mpatches.Patch(color=[0.2,0.5,1.0],    label='Target region'),
        mpatches.Patch(color='#ff4444',         label='Agent'),
    ]
    axes[0].legend(handles=legend, loc='upper right', fontsize=7,
                   facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    trajectory = [tuple(env.pos)]
    frames = []
    step_data = {'step': 0, 'reward': 0.0, 'coverage': 0.0, 'score': 0.0, 'done': False}

    def update(_frame):
        if step_data['done']:
            return

        action, _, _, _ = trainer.act(obs, deterministic=False)
        new_obs, reward, terminated, truncated, info = env.step(action)
        obs[:] = new_obs
        trajectory.append(tuple(env.pos))

        step_data['step'] += 1
        step_data['reward'] += reward
        step_data['coverage'] = info['region_coverage']
        step_data['score'] = info['region_score']
        step_data['done'] = terminated or truncated

        # Update SLAM panel
        si = slam_to_image(env.slam.binary_map, W, H)
        si = overlay_region(si, region_mask)
        si = overlay_trajectory(si, trajectory[-80:], W, H)
        slam_disp.set_data(si)
        agent_dot_slam.set_data([env.pos[0]], [env.pos[1]])

        # Update truth panel
        ti = overlay_region(truth_img.copy(), region_mask)
        ti = overlay_trajectory(ti, trajectory[-80:], W, H)
        truth_disp.set_data(ti)
        agent_dot_truth.set_data([env.pos[0]], [env.pos[1]])

        stats_text.set_text(
            f"Step {step_data['step']:4d} / {args.max_steps}  |  "
            f"Coverage {step_data['coverage']*100:.1f}%  |  "
            f"Score {step_data['score']*100:.1f}%  |  "
            f"Reward {step_data['reward']:.1f}"
        )

        if step_data['done']:
            result = 'SUCCESS' if terminated else 'TIMEOUT'
            fig.suptitle(
                f"{result} — {step_data['coverage']*100:.1f}% covered",
                color='#44ff88' if terminated else '#ffaa44', fontsize=13
            )

        if args.speed > 0:
            plt.pause(args.speed)

        return slam_disp, truth_disp, agent_dot_slam, agent_dot_truth, stats_text

    if args.save_gif:
        print("Recording GIF (this will take a while)...")
        ani = FuncAnimation(fig, update, frames=args.max_steps,
                            interval=80, blit=False, repeat=False)
        out_path = Path(args.out)
        out_path.parent.mkdir(exist_ok=True)
        ani.save(str(out_path), writer=PillowWriter(fps=12))
        print(f"Saved: {out_path}")
    else:
        ani = FuncAnimation(fig, update, frames=args.max_steps,
                            interval=max(1, int(args.speed * 1000)),
                            blit=False, repeat=False)
        plt.show()

    print(f"\nFinal — Steps: {step_data['step']} | "
          f"Coverage: {step_data['coverage']*100:.1f}% | "
          f"Score: {step_data['score']*100:.1f}%")


if __name__ == '__main__':
    args = parse_args()
    run_visualization(args)
