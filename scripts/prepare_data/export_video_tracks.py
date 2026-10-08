"""Convert AllTracker dense absolute pixel trajectories into Video loader TIFFs.

Input NPZ: forward_tracks/backward_tracks [T,2,H,W] absolute (x,y),
forward_confidence/backward_confidence [T,1,H,W] in [0,1].
Forward tracks start at the first video frame, backward tracks at the last.
Run tracking on frames at the chosen stride (default 8), independently per direction.
Keep the original full-rate MP4 alongside its `<stem>__supp` output directory.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tifffile


def export_tracks(npz_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    with np.load(npz_path, allow_pickle=False) as arrays:
        for name, prefix, field in [('forward', 'fwd', 'traj_maps_e'), ('backward', 'bwd', 'bwd_traj_maps_e')]:
            tracks = arrays[name + '_tracks'].astype(np.float32)
            conf = arrays[name + '_confidence'].astype(np.float32)
            if tracks.ndim != 4 or tracks.shape[1] != 2 or conf.shape != (len(tracks), 1, *tracks.shape[2:]):
                raise ValueError('Expected tracks [T,2,H,W], confidence [T,1,H,W]')
            if len(tracks) < 5 or not np.isfinite(tracks).all() or not np.isfinite(conf).all():
                raise ValueError('Need at least five finite track frames per direction')
            lo, hi = float(tracks.min()), float(tracks.max())
            if hi <= lo:
                raise ValueError('Degenerate trajectory range')
            metadata[field + '_min'], metadata[field + '_max'] = lo, hi
            quantized = np.round(((tracks - lo) / (hi - lo) * 2 - 1) * 32767).astype(np.int16)
            confidence = np.round(np.clip(conf, 0, 1) * 255).astype(np.uint8)
            for t in range(len(tracks)):
                with tifffile.TiffWriter(output / f'{prefix}-{t:03d}.tiff') as writer:
                    writer.write(quantized[t, 0], photometric='minisblack')
                    writer.write(quantized[t, 1], photometric='minisblack')
                    writer.write(confidence[t, 0], photometric='minisblack')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    export_tracks(args.tracks, args.output)
