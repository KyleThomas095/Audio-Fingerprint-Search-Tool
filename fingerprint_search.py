# fingerprint_search.py
"""
Audio Fingerprint Search Tool

Index local audio files and find matches by presenting audio samples —
either from files or captured directly from system audio.

Uses Chromaprint for perceptual fingerprinting, meaning compressed formats
(mp3) will match their source files (wav) despite encoding differences.

Repository: https://github.com/KyleThomas095/Audio-Fingerprint-Search-Tool
License: CC0
"""

import argparse
import hashlib
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from tqdm import tqdm


# === Configuration ===
DB_PATH = Path.home() / ".audio_fingerprints.db"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_THRESHOLD = 0.4
DEFAULT_RECORD_DURATION = 8


# === Database Layer ===

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database for fingerprint storage."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            id INTEGER PRIMARY KEY,
            filepath TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            duration REAL,
            file_hash TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_filename ON fingerprints(filename)")
    conn.commit()
    return conn


def store_fingerprint(conn: sqlite3.Connection, filepath: str, fingerprint: str,
                      duration: float, file_hash: str):
    """Store a fingerprint in the database."""
    filename = os.path.basename(filepath)
    conn.execute("""
        INSERT OR REPLACE INTO fingerprints (filepath, filename, fingerprint, duration, file_hash)
        VALUES (?, ?, ?, ?, ?)
    """, (filepath, filename, fingerprint, duration, file_hash))
    conn.commit()


def get_all_fingerprints(conn: sqlite3.Connection) -> list[tuple]:
    """Retrieve all fingerprints for matching."""
    cursor = conn.execute("SELECT filepath, filename, fingerprint FROM fingerprints")
    return cursor.fetchall()


def get_indexed_count(conn: sqlite3.Connection) -> int:
    """Get count of indexed files."""
    cursor = conn.execute("SELECT COUNT(*) FROM fingerprints")
    return cursor.fetchone()[0]


# === Fingerprint Generation ===

def get_chromaprint(audio_path: str, duration: int = 120) -> Optional[tuple[str, float]]:
    """
    Generate chromaprint fingerprint using fpcalc.
    
    Args:
        audio_path: Path to audio file
        duration: Maximum seconds to analyze (default 120)
    
    Returns:
        Tuple of (fingerprint_string, duration) or None on failure
    """
    try:
        result = subprocess.run(
            ["fpcalc", "-raw", "-length", str(duration), audio_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return None

        fingerprint = None
        file_duration = 0.0

        for line in result.stdout.strip().split("\n"):
            if line.startswith("FINGERPRINT="):
                fingerprint = line.split("=", 1)[1]
            elif line.startswith("DURATION="):
                file_duration = float(line.split("=", 1)[1])

        if fingerprint:
            return fingerprint, file_duration
        return None

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def file_hash(filepath: str, chunk_size: int = 8192) -> str:
    """Generate MD5 hash of file for change detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# === Fingerprint Comparison ===

def parse_raw_fingerprint(fp_str: str) -> np.ndarray:
    """Convert comma-separated fingerprint string to numpy array."""
    return np.array([int(x) for x in fp_str.split(",")], dtype=np.uint32)


def segment_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Calculate similarity between two equal-length fingerprint segments.
    
    Uses bitwise XOR and popcount to measure perceptual difference.
    
    Returns:
        Score from 0.0 (no match) to 1.0 (identical)
    """
    if len(fp1) != len(fp2) or len(fp1) == 0:
        return 0.0

    xor_result = np.bitwise_xor(fp1, fp2)
    total_bits = len(fp1) * 32
    differing_bits = sum(bin(int(x)).count('1') for x in xor_result)

    similarity = 1.0 - (differing_bits / total_bits)
    return max(0.0, similarity)


def fingerprint_similarity(query_fp: np.ndarray, stored_fp: np.ndarray) -> float:
    """
    Sliding window comparison to find best alignment of query within stored.
    
    This handles the case where a sample is taken from the middle of a track —
    the query fingerprint is slid along the stored fingerprint to find the
    position with maximum similarity.
    
    Returns:
        Best similarity score across all positions
    """
    query_len = len(query_fp)
    stored_len = len(stored_fp)

    if query_len == 0 or stored_len == 0:
        return 0.0

    if query_len > stored_len:
        query_fp, stored_fp = stored_fp, query_fp
        query_len, stored_len = stored_len, query_len

    best_score = 0.0
    step = max(1, query_len // 10)  # ~0.5 second steps for speed

    for offset in range(0, stored_len - query_len + 1, step):
        segment = stored_fp[offset:offset + query_len]
        score = segment_similarity(query_fp, segment)
        if score > best_score:
            best_score = score
            if best_score > 0.85:  # Early exit on strong match
                return best_score

    return best_score


def find_matches(conn: sqlite3.Connection, query_fp: str,
                 threshold: float = DEFAULT_THRESHOLD, 
                 top_n: int = 5) -> list[tuple[str, str, float]]:
    """
    Find matching files for a query fingerprint.
    
    Args:
        conn: Database connection
        query_fp: Raw fingerprint string from fpcalc
        threshold: Minimum similarity score (0.0-1.0)
        top_n: Maximum results to return
    
    Returns:
        List of (filepath, filename, similarity_score) tuples
    """
    query_array = parse_raw_fingerprint(query_fp)
    all_fps = get_all_fingerprints(conn)

    matches = []
    for filepath, filename, stored_fp in tqdm(all_fps, desc="Comparing", unit="files"):
        try:
            stored_array = parse_raw_fingerprint(stored_fp)
            similarity = fingerprint_similarity(query_array, stored_array)
            if similarity >= threshold:
                matches.append((filepath, filename, similarity))
        except Exception:
            continue

    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:top_n]


# === Audio Capture ===

def list_audio_devices():
    """List available audio devices for recording."""
    print("\n=== Audio Devices ===\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        direction = []
        if dev['max_input_channels'] > 0:
            direction.append(f"IN:{dev['max_input_channels']}ch")
        if dev['max_output_channels'] > 0:
            direction.append(f"OUT:{dev['max_output_channels']}ch")

        marker = ""
        if i == sd.default.device[0]:
            marker += " [DEFAULT INPUT]"
        if i == sd.default.device[1]:
            marker += " [DEFAULT OUTPUT]"

        print(f"  {i}: {dev['name']} ({', '.join(direction)}){marker}")

    print("\n=== PulseAudio/PipeWire Sources ===\n")
    try:
        result = subprocess.run(["pactl", "list", "sources", "short"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    source_id, source_name = parts[0], parts[1]
                    is_monitor = ".monitor" in source_name
                    print(f"  {source_id}: {source_name}" + 
                          (" [MONITOR - for system audio]" if is_monitor else ""))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  (pactl not available)")

    print()


def record_audio(duration: float, device: Optional[int] = None,
                 sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Record audio from microphone using sounddevice."""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=2,
        device=device,
        dtype=np.float32
    )
    sd.wait()
    print("Recording complete.")
    return recording


def save_temp_wav(audio_data: np.ndarray, 
                  sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    """Save audio data to a temporary WAV file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(temp_file.name, sample_rate, audio_int16)
    return temp_file.name


def get_monitor_device() -> Optional[int]:
    """Try to find a monitor source for system audio capture via sounddevice."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name_lower = dev['name'].lower()
        if 'monitor' in name_lower and dev['max_input_channels'] > 0:
            return i
    return None


def get_pactl_monitor_source() -> Optional[str]:
    """Find a monitor source using pactl (PipeWire/PulseAudio)."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        # First pass: prefer USB/external monitor (likely primary audio device)
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                source_name = parts[1]
                if ".monitor" in source_name:
                    if "usb" in source_name.lower():
                        return source_name

        # Second pass: any monitor source
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2 and ".monitor" in parts[1]:
                return parts[1]

        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def record_with_parec(duration: float, source: str,
                      sample_rate: int = DEFAULT_SAMPLE_RATE) -> Optional[str]:
    """
    Record from a PulseAudio/PipeWire monitor source.
    
    Uses parec for raw PCM capture, piped through ffmpeg to create a proper
    WAV file. This is the most reliable method for capturing system audio
    on Linux with PipeWire.
    
    Args:
        duration: Seconds to record
        source: PulseAudio source name (from `pactl list sources short`)
        sample_rate: Sample rate in Hz
    
    Returns:
        Path to temporary WAV file, or None on failure
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        print(f"Recording {duration}s from: {source}")

        parec_cmd = [
            "parec",
            "-d", source,
            f"--rate={sample_rate}",
            "--channels=2",
            "--format=s16le"
        ]

        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", "2",
            "-i", "pipe:0",
            "-y",
            temp_path
        ]

        parec_proc = subprocess.Popen(parec_cmd, stdout=subprocess.PIPE, 
                                       stderr=subprocess.DEVNULL)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=parec_proc.stdout,
                                        stdout=subprocess.DEVNULL, 
                                        stderr=subprocess.DEVNULL)
        parec_proc.stdout.close()

        try:
            ffmpeg_proc.wait(timeout=duration + 2)
        except subprocess.TimeoutExpired:
            pass
        finally:
            parec_proc.terminate()
            ffmpeg_proc.terminate()
            parec_proc.wait()
            ffmpeg_proc.wait()

    except FileNotFoundError as e:
        print(f"Error: Required tool not found: {e}")
        print("Install with: sudo apt install pulseaudio-utils ffmpeg")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return None

    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
        print("Recording complete.")
        return temp_path
    else:
        print("Recording failed or empty.")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return None


# === Indexing ===

def index_directory(directory: str, conn: sqlite3.Connection):
    """
    Index all audio files in a directory.
    
    Generates chromaprint fingerprints for each supported audio file
    and stores them in the database for later matching.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)

    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(dir_path.glob(f"*{ext}"))
        audio_files.extend(dir_path.glob(f"*{ext.upper()}"))

    audio_files = sorted(set(audio_files))
    print(f"Found {len(audio_files)} audio files to index.\n")

    if not audio_files:
        return

    indexed = 0
    failed = 0

    for audio_file in tqdm(audio_files, desc="Indexing", unit="files"):
        filepath = str(audio_file.absolute())

        try:
            result = get_chromaprint(filepath)
            if result:
                fingerprint, duration = result
                fhash = file_hash(filepath)
                store_fingerprint(conn, filepath, fingerprint, duration, fhash)
                indexed += 1
            else:
                failed += 1
        except Exception as e:
            tqdm.write(f"Failed: {audio_file.name} - {e}")
            failed += 1

    print(f"\nIndexing complete: {indexed} succeeded, {failed} failed")
    print(f"Total files in database: {get_indexed_count(conn)}")


# === Search ===

def search_from_file(filepath: str, conn: sqlite3.Connection, threshold: float):
    """Search for matches using an audio file as the query."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    print(f"Generating fingerprint for: {filepath}")
    result = get_chromaprint(filepath)

    if not result:
        print("Failed to generate fingerprint from file.")
        sys.exit(1)

    fingerprint, duration = result
    print(f"Sample duration: {duration:.1f}s\n")

    run_search(conn, fingerprint, threshold)


def search_from_recording(duration: float, conn: sqlite3.Connection,
                          threshold: float, use_monitor: bool = False):
    """Search for matches by recording audio (mic or system)."""
    temp_wav = None

    if use_monitor:
        source = get_pactl_monitor_source()
        if source is None:
            print("Error: Could not find monitor source.")
            print("Run 'pactl list sources short' to check available sources.")
            return

        temp_wav = record_with_parec(duration, source)
        if temp_wav is None:
            return
    else:
        audio_data = record_audio(duration, device=None)
        temp_wav = save_temp_wav(audio_data)

    try:
        print(f"Processing recorded audio...")
        result = get_chromaprint(temp_wav)

        if not result:
            print("Failed to generate fingerprint from recording.")
            return

        fingerprint, _ = result
        run_search(conn, fingerprint, threshold)

    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)


def run_search(conn: sqlite3.Connection, fingerprint: str, threshold: float):
    """Execute search and display results."""
    print("Searching for matches...\n")
    matches = find_matches(conn, fingerprint, threshold=threshold)

    if not matches:
        print("No matches found above threshold.")
        return

    print("=== Matches Found ===\n")
    for i, (filepath, filename, score) in enumerate(matches, 1):
        confidence = "HIGH" if score > 0.6 else "MEDIUM" if score > 0.5 else "LOW"
        print(f"{i}. [{confidence}] {score:.1%} match")
        print(f"   File: {filename}")
        print(f"   Path: {filepath}\n")


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="Audio Fingerprint Search Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Index a directory:
    %(prog)s index /path/to/music

  Search using an audio file:
    %(prog)s search --file snippet.wav

  Search by recording from microphone:
    %(prog)s search --record 8

  Search by capturing system audio:
    %(prog)s search --monitor 8

  List audio devices:
    %(prog)s list-devices
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser(
        "index", 
        help="Index audio files in a directory"
    )
    index_parser.add_argument("directory", help="Directory containing audio files")

    # Search command
    search_parser = subparsers.add_parser(
        "search", 
        help="Search for matching audio"
    )
    search_group = search_parser.add_mutually_exclusive_group(required=True)
    search_group.add_argument(
        "--file", "-f",
        help="Audio file to search for"
    )
    search_group.add_argument(
        "--record", "-r", 
        type=float,
        metavar="SECONDS",
        help="Record from microphone for N seconds"
    )
    search_group.add_argument(
        "--monitor", "-m", 
        type=float,
        metavar="SECONDS",
        help="Record from system audio for N seconds"
    )
    search_parser.add_argument(
        "--threshold", "-t", 
        type=float, 
        default=DEFAULT_THRESHOLD,
        help=f"Match threshold 0.0-1.0 (default: {DEFAULT_THRESHOLD})"
    )

    # List devices command
    subparsers.add_parser(
        "list-devices", 
        help="List available audio devices"
    )

    # Stats command
    subparsers.add_parser(
        "stats", 
        help="Show database statistics"
    )

    args = parser.parse_args()

    if args.command == "list-devices":
        list_audio_devices()
        return

    conn = init_db()

    if args.command == "index":
        index_directory(args.directory, conn)

    elif args.command == "search":
        if get_indexed_count(conn) == 0:
            print("Error: No files indexed yet. Run 'index' command first.")
            sys.exit(1)

        if args.file:
            search_from_file(args.file, conn, args.threshold)
        elif args.record:
            search_from_recording(args.record, conn, args.threshold, use_monitor=False)
        elif args.monitor:
            search_from_recording(args.monitor, conn, args.threshold, use_monitor=True)

    elif args.command == "stats":
        count = get_indexed_count(conn)
        print(f"Database: {DB_PATH}")
        print(f"Indexed files: {count}")

    conn.close()


if __name__ == "__main__":
    main()
