from __future__ import annotations
from pathlib import Path
from typing import Iterable
import imageio.v2 as imageio
from PIL import Image, ImageOps
import numpy as np


def build_sequence_video(image_paths: Iterable[str | Path], out_path: str | Path, fps: int = 2) -> Path:
    """Build video of daily LCS/accumulation MAP CHANGES, not particle advection."""
    out = Path(out_path)
    paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError('No output-day images available for MP4')
    out.parent.mkdir(parents=True, exist_ok=True)
    size=(1280, 960)
    with imageio.get_writer(out, fps=max(1,int(fps)), codec='libx264', macro_block_size=16) as writer:
        for path in paths:
            with Image.open(path) as im:
                frame=ImageOps.pad(im.convert('RGB'), size, color=(10,20,35))
                writer.append_data(np.asarray(frame))
    return out
