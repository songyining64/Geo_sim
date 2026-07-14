"""Merge independently completed blocked-comparison seed outputs."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.benchmark_blocked_comparison import aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+', help='Input blocked_comparison.json files')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()

    merged = []
    configs = []
    for filename in args.results:
        with open(filename, 'r') as f:
            payload = json.load(f)
        configs.append(payload['config'])
        by_steps = payload.get('per_seed_by_adapt_steps', {})
        if '1' in by_steps:
            merged.extend(by_steps['1'])
        else:
            merged.extend(payload['per_seed'])

    merged.sort(key=lambda item: item['seed'])
    seeds = [item['seed'] for item in merged]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f'duplicate seeds in inputs: {seeds}')

    payload = {
        'config': {**configs[0], 'seeds': seeds},
        'per_seed': merged,
        'aggregate': aggregate(merged),
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'blocked_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload['aggregate'], indent=2))
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
