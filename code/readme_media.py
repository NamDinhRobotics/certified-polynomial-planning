"""Extract README previews from the shipped movie; no simulation is performed."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "video" / "v3_execution.mp4"
CLIPS = (
    ("v3-replay.gif", 0, 15, 960, 10, 256, "V3 scene 41013, conic arm, repetition 0"),
    ("quadrotor-illustration.gif", 49, 8, 640, 8, 96,
     "Earlier affine-guide flight illustration, outside V3"),
)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(ffmpeg, *args):
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("Install FFmpeg and ensure ffmpeg is on PATH.")
    if not SOURCE.is_file():
        parser.error(f"Missing source movie: {SOURCE}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "video/v3_execution.mp4",
        "source_sha256": sha256(SOURCE),
        "scope": "Excerpts of the existing movie; no new planning or dynamics run.",
        "ffmpeg_version": subprocess.check_output(
            [ffmpeg, "-version"], text=True
        ).splitlines()[0],
        "outputs": [],
    }
    for filename, start, duration, width, fps, colors, scope in CLIPS:
        output = out / filename
        print(f"Creating {filename}", flush=True)
        run(
            ffmpeg, "-ss", start, "-t", duration, "-i", SOURCE,
            "-filter_complex",
            f"[0:v]fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
            f"[a]palettegen=stats_mode=diff:max_colors={colors}[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle[v]",
            "-map", "[v]", "-an", "-loop", "0", output,
        )
        manifest["outputs"].append({
            "file": filename, "start_seconds": start, "duration_seconds": duration,
            "fps": fps, "width": width, "palette_colors": colors,
            "playback_speed": 1, "scope": scope,
            "bytes": output.stat().st_size, "sha256": sha256(output),
        })
    output = out / "v3-replay.mp4"
    print(f"Creating {output.name}", flush=True)
    run(
        ffmpeg, "-i", SOURCE, "-t", 15, "-map", "0:v:0", "-an",
        "-vf", "scale=1280:-2", "-c:v", "libx264", "-crf", 20,
        "-preset", "medium", "-pix_fmt", "yuv420p", "-map_metadata", "-1",
        "-movflags", "+faststart", output,
    )
    manifest["outputs"].append({
        "file": output.name, "start_seconds": 0, "duration_seconds": 15,
        "scope": CLIPS[0][-1], "bytes": output.stat().st_size,
        "sha256": sha256(output),
    })
    (out / "media-provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Media and provenance written to {out}")


if __name__ == "__main__":
    main()
