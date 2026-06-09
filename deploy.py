"""
deploy.py — Run the SLAM agent on a real map and save a GIF
------------------------------------------------------------
Usage:
    python deploy.py --map data/real_grid.json
    python deploy.py --map data/real_grid.json --lat 42.361 --lon -71.092
    python deploy.py --map data/real_grid.json --lat 42.361 --lon -71.092 --radius 10
"""

import argparse
import json
import subprocess
import sys


def gps_to_grid(lat, lon, city):
    """Convert GPS coordinates to grid cell (x, y)."""
    x = int((lon - city['west'])  / (city['east']  - city['west'])  * city['W'])
    y = int((city['north'] - lat) / (city['north'] - city['south']) * city['H'])
    x = max(0, min(city['W'] - 1, x))
    y = max(0, min(city['H'] - 1, y))
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map',      default='data/real_grid.json')
    p.add_argument('--lat',      type=float, default=None, help='Target latitude')
    p.add_argument('--lon',      type=float, default=None, help='Target longitude')
    p.add_argument('--radius',   type=int,   default=8,    help='Target radius in grid cells')
    p.add_argument('--steps',    type=int,   default=1200)
    p.add_argument('--checkpoint', default='checkpoints/mamba_fast_hybrid_slam.pt')
    p.add_argument('--out',      default='evaluation_results/agent_run.gif')
    args = p.parse_args()

    with open(args.map) as f:
        city = json.load(f)

    print(f"Map:  {city.get('name', args.map)}  ({city['W']}×{city['H']}, {city.get('cell_size',10)}m/cell)")

    if args.lat is not None and args.lon is not None:
        tx, ty = gps_to_grid(args.lat, args.lon, city)
        print(f"Target GPS ({args.lat}, {args.lon}) → grid ({tx}, {ty})")
    elif 'target' in city:
        tx, ty = city['target']
        print(f"Target: map default {[tx, ty]}")
    else:
        tx, ty = city['W'] // 2, city['H'] // 2
        print(f"Target: map centre ({tx}, {ty})")

    print(f"Radius: {args.radius} cells  (~{args.radius * city.get('cell_size', 10)}m)")
    print(f"Output: {args.out}\n")

    cmd = [
        sys.executable, 'visualize_agent.py',
        '--checkpoint', args.checkpoint,
        '--city-map',   args.map,
        '--target-x',   str(tx),
        '--target-y',   str(ty),
        '--target-radius', str(args.radius),
        '--max-steps',  str(args.steps),
        '--speed',      '0',
        '--save-gif',
    ]

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\nDone — open {args.out} to watch the run.")
    else:
        print("\nSomething went wrong — check the output above.")


if __name__ == '__main__':
    main()
