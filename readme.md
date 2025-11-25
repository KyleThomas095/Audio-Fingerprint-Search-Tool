# Audio Fingerprint Search

A local acoustic fingerprinting tool for finding files by playing audio samples. Index your audio library once, then identify tracks by playing a snippet — from a file, microphone, or system audio capture.

Uses [Chromaprint](https://acoustid.org/chromaprint) for perceptual fingerprinting, meaning compressed formats (mp3) will match their source files (wav) despite encoding differences.

## Use Cases

- **Library management**: Find the source file for a track playing on a streaming service
- **Deduplication**: Identify same-content files across different formats (wav/mp3/flac)
- **Audio archaeology**: Match snippets against large collections of AI-generated music, samples, or recordings

## Requirements

**System dependencies** (Ubuntu/Debian):

```bash
sudo apt install ffmpeg libchromaprint-tools pulseaudio-utils
```

**Python 3.10+** with:

```bash
pip install numpy sounddevice scipy tqdm
```

## Installation

```bash
git clone https://github.com/KyleThomas095/Audio-Fingerprint-Search-Tool.git
cd audio-fingerprint-search

python3 -m venv venv
source venv/bin/activate
pip install numpy sounddevice scipy tqdm
```

## Usage

### Index your library

```bash
python fingerprint_search.py index /path/to/audio/files
```

This scans all supported audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) and stores fingerprints in `~/.audio_fingerprints.db`.

Indexing ~5000 files takes approximately 15-30 minutes depending on file lengths and disk speed.

### Search by file

```bash
python fingerprint_search.py search --file snippet.wav
```

### Search by recording (microphone)

```bash
python fingerprint_search.py search --record 8
```

Records 8 seconds from your default microphone input.

### Search by system audio capture

```bash
python fingerprint_search.py search --monitor 8
```

Captures 8 seconds of whatever is currently playing through your speakers. Requires PipeWire or PulseAudio (standard on modern Ubuntu).

### List audio devices

```bash
python fingerprint_search.py list-devices
```

Shows available audio inputs and PulseAudio/PipeWire sources.

### Check database stats

```bash
python fingerprint_search.py stats
```

## Options

| Option | Description |
|--------|-------------|
| `--threshold`, `-t` | Match threshold 0.0-1.0 (default: 0.4). Higher = stricter matching. |
| `--file`, `-f` | Path to audio file to search for |
| `--record`, `-r` | Record from microphone for N seconds |
| `--monitor`, `-m` | Record from system audio for N seconds |

## How It Works

1. **Fingerprinting**: Chromaprint analyzes audio and produces a sequence of 32-bit integers representing perceptual features — which frequency bands are dominant relative to neighbors, roughly every 0.1 seconds.

2. **Storage**: Fingerprints are stored in SQLite with file paths and metadata.

3. **Matching**: Uses sliding window comparison — the query fingerprint is slid along each stored fingerprint to find the best alignment. This handles samples taken from any point in a track, not just the beginning.

4. **Scoring**: Bit-level XOR comparison counts how many bits differ between aligned segments. >60% similarity is typically a strong match; >85% is near-certain.

## Sample Output

```
=== Matches Found ===

1. [HIGH] 99.3% match
   File: 0251_suno_where_is_the_sound_coming_from.wav
   Path: /home/user/music/library/0251_suno_where_is_the_sound_coming_from.wav

2. [HIGH] 71.9% match
   File: 0942_recording_cover_remix.wav
   Path: /home/user/music/library/0942_recording_cover_remix.wav
```

## Recording Duration

Longer samples (8-12 seconds) are recommended:

- More distinctive patterns reduce false positives
- Actually searches faster (fewer sliding window positions)
- Better handles repetitive sections (intros, loops)

Short samples (3-5 seconds) work but may match multiple similar tracks.

## Troubleshooting

### System audio capture fails

Check available monitor sources:

```bash
pactl list sources short
```

Look for sources ending in `.monitor`. If using an external audio interface (like Scarlett 2i2), ensure audio is routed through it.

### No matches found

- Verify files are indexed: `python fingerprint_search.py stats`
- Try lowering threshold: `--threshold 0.3`
- Ensure sample has actual audio content (not silence)

### fpcalc not found

```bash
sudo apt install libchromaprint-tools
```

## License

CC0

## Acknowledgments

- [Chromaprint](https://acoustid.org/chromaprint) by Lukáš Lalinský
- Developed with assistance from Claude (Anthropic)
