#!/usr/bin/env python3
"""Generate tiered real-life .npz datasets for regex.

Input format: 8 basis bit-plane streams (b0..b7, Parabix byte decomposition).
Total = 8 * N_bytes / 8 = N_bytes (payload bytes from pcap).

Tiers:
  small:  1M bytes   (~1 MB)
  medium: 10M bytes  (~10 MB)
  large:  100M bytes (~100 MB)

The small tier also emits wrccdc2012_10k.npz (first 10,000 payload bytes,
a prefix of the 1M file) so the optional generated unit test registered in
datasets/tests/ is reproducible from this script.

Source: WRCCDC 2012 network capture (wrccdc2012.pcap).
        File must be placed in raw/ directory.
        Download: https://archive.wrccdc.org/pcaps/2012/wrccdc2012.pcap.gz

The pcap file contains ~28 MB of raw data. For the large tier (100 MB),
if insufficient payload bytes exist, available data is tiled to reach
the target.

Usage:
    python make_data.py                   # Generate all tiers
    python make_data.py --tier small      # Generate one tier
"""

import argparse
import os
import struct
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
PCAP_FILE = os.path.join(DIR, "raw", "wrccdc2012.pcap")
PCAP_FILE_GZ = PCAP_FILE + ".gz"

TIERS = {
    "small":  1_000_000,     # 1M bytes -> ~1 MB
    "medium": 10_000_000,    # 10M bytes -> ~10 MB
    "large":  100_000_000,   # 100M bytes -> ~100 MB
}

OUTPUT_NAMES = {
    "small":  "wrccdc2012_1m.npz",
    "medium": "wrccdc2012_10m.npz",
    "large":  "wrccdc2012_100m.npz",
}

# The 10K dataset is the first 10,000 payload bytes (a prefix of the
# small-tier 1M file).  datasets/tests/ registers an optional generated
# unit test against it (run.py: DATASET_PATHS["wrccdc2012_10k"]); it is
# emitted alongside the small tier so the documented pipeline reproduces
# every file the test registry references.
PREFIX_10K_NAME = "wrccdc2012_10k.npz"
PREFIX_10K_BYTES = 10_000


def extract_pcap_payload(pcap_path: str, max_bytes: int | None = None) -> bytes:
    """Extract TCP/UDP payload bytes from pcap file (raw parsing, no scapy)."""
    with open(pcap_path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic == 0xa1b2c3d4:
            endian = '<'
        elif magic == 0xd4c3b2a1:
            endian = '>'
        else:
            raise ValueError(f"Unknown pcap magic: {magic:#x}")

        f.read(20)  # skip rest of global header

        payloads = []
        total_bytes = 0

        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            _, _, incl_len, _ = struct.unpack(endian + 'IIII', hdr)
            pkt_data = f.read(incl_len)
            if len(pkt_data) < incl_len:
                break

            # Parse Ethernet + IP + TCP/UDP
            if len(pkt_data) >= 34:
                eth_type = struct.unpack('!H', pkt_data[12:14])[0]
                if eth_type == 0x0800:  # IPv4
                    ip_start = 14
                    if ip_start + 20 <= len(pkt_data):
                        ihl = (pkt_data[ip_start] & 0x0f) * 4
                        ip_proto = pkt_data[ip_start + 9]
                        ip_total_len = struct.unpack('!H',
                            pkt_data[ip_start+2:ip_start+4])[0]

                        payload_data = None
                        if ip_proto == 6 and ip_start + ihl + 20 <= len(pkt_data):
                            tcp_start = ip_start + ihl
                            tcp_doff = (pkt_data[tcp_start + 12] >> 4) * 4
                            p_start = tcp_start + tcp_doff
                            p_end = min(ip_start + ip_total_len, len(pkt_data))
                            if p_start < p_end:
                                payload_data = pkt_data[p_start:p_end]
                        elif ip_proto == 17 and ip_start + ihl + 8 <= len(pkt_data):
                            udp_start = ip_start + ihl
                            udp_len = struct.unpack('!H',
                                pkt_data[udp_start+4:udp_start+6])[0]
                            p_start = udp_start + 8
                            p_end = min(udp_start + udp_len, len(pkt_data))
                            if p_start < p_end:
                                payload_data = pkt_data[p_start:p_end]

                        if payload_data:
                            payloads.append(payload_data)
                            total_bytes += len(payload_data)
                            if max_bytes and total_bytes >= max_bytes:
                                break

    result = b''.join(payloads)
    if max_bytes:
        result = result[:max_bytes]
    return result


def bytes_to_basis_npz(payload: bytes, npz_path: str,
                       source_bytes: int | None = None,
                       tile_factor: int = 1):
    """Transpose byte array to 8 basis bit-planes and save as .npz.

    ``source_bytes`` records the count of *unique* payload bytes available
    from the pcap before tiling; ``tile_factor`` is how many times that
    block was repeated to reach ``len(payload)``.  Both are stored in the
    .npz so downstream tools can warn about the periodic-content artefact.
    """
    n = len(payload)
    arr = np.frombuffer(payload, dtype=np.uint8)

    planes = {}
    for bit in range(8):
        plane = ((arr >> bit) & 1).astype(np.uint8)
        planes[f"b{bit}"] = np.packbits(plane, bitorder='little')

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    extra = {}
    if source_bytes is not None:
        extra["source_bytes"] = source_bytes
    if tile_factor != 1:
        extra["tile_factor"] = tile_factor
    np.savez_compressed(npz_path, payload_length=n, **planes, **extra)


def generate_tier(tier: str):
    """Generate one tier of regex data."""
    max_bytes = TIERS[tier]
    out_dir = os.path.join(DIR, tier)
    out_path = os.path.join(out_dir, OUTPUT_NAMES[tier])

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  [{tier}] Already exists: {out_path} ({size:,} bytes) -- skipping")
        return

    pcap_path = PCAP_FILE
    if not os.path.isfile(pcap_path):
        # Try decompressing .gz if available
        if os.path.isfile(PCAP_FILE_GZ):
            import gzip, shutil
            print(f"  [{tier}] Decompressing {PCAP_FILE_GZ}...")
            with gzip.open(PCAP_FILE_GZ, 'rb') as f_in:
                with open(PCAP_FILE, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"  [{tier}] SKIP: pcap file not found: {PCAP_FILE}")
            print(f"         Download from: https://archive.wrccdc.org/pcaps/2012/wrccdc2012.pcap.gz")
            return

    print(f"  [{tier}] Extracting up to {max_bytes:,} payload bytes from pcap...")
    payload = extract_pcap_payload(PCAP_FILE, max_bytes=None)  # Get all available
    available = len(payload)
    print(f"  [{tier}] Available payload: {available:,} bytes")

    tile_factor = 1
    if available < max_bytes:
        # Tile payload to reach target.  Tiling introduces *periodic* content
        # that boosts match counts for any pattern that hits inside the
        # repeated block.  Loud-warn so downstream consumers know.
        tile_factor = (max_bytes + available - 1) // available
        print(f"  [{tier}] WARNING: tiling payload {tile_factor}x to reach "
              f"{max_bytes:,} bytes ({available:,} unique → {tile_factor} copies)",
              file=sys.stderr)
        payload = (payload * tile_factor)[:max_bytes]
    else:
        payload = payload[:max_bytes]

    print(f"  [{tier}] Using {len(payload):,} payload bytes "
          f"(unique={available:,}, tile_factor={tile_factor})")
    bytes_to_basis_npz(payload, out_path,
                       source_bytes=available, tile_factor=tile_factor)
    size = os.path.getsize(out_path)
    print(f"  [{tier}] Wrote {out_path} ({size:,} bytes)")


def _ensure_pcap(tag: str) -> bool:
    """Make sure raw/wrccdc2012.pcap exists (decompress .gz if needed).

    Returns True if the pcap is available, False otherwise.
    """
    if os.path.isfile(PCAP_FILE):
        return True
    if os.path.isfile(PCAP_FILE_GZ):
        import gzip, shutil
        print(f"  [{tag}] Decompressing {PCAP_FILE_GZ}...")
        with gzip.open(PCAP_FILE_GZ, 'rb') as f_in:
            with open(PCAP_FILE, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    print(f"  [{tag}] SKIP: pcap file not found: {PCAP_FILE}")
    print(f"         Download from: https://archive.wrccdc.org/pcaps/2012/wrccdc2012.pcap.gz")
    return False


def generate_small_prefix():
    """Write small/wrccdc2012_10k.npz (first 10,000 pcap payload bytes).

    This is exactly the first PREFIX_10K_BYTES bytes of the small-tier
    payload (pcap payload concatenated in capture order), so it is a
    deterministic prefix of wrccdc2012_1m.npz and reproduces the file
    committed for the optional generated unit test.
    """
    out_path = os.path.join(DIR, "small", PREFIX_10K_NAME)
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  [10k] Already exists: {out_path} ({size:,} bytes) -- skipping")
        return
    if not _ensure_pcap("10k"):
        return
    payload = extract_pcap_payload(PCAP_FILE, max_bytes=PREFIX_10K_BYTES)
    print(f"  [10k] Extracted {len(payload):,} payload bytes")
    bytes_to_basis_npz(payload, out_path)
    size = os.path.getsize(out_path)
    print(f"  [10k] Wrote {out_path} ({size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Generate regex tiered datasets")
    parser.add_argument("--tier", choices=["small", "medium", "large"],
                        help="Generate only this tier (default: all)")
    args = parser.parse_args()

    print("regex: generating tiered datasets from WRCCDC 2012 pcap")
    print(f"  Source: {PCAP_FILE}")
    print()

    tiers = [args.tier] if args.tier else ["small", "medium", "large"]
    for tier in tiers:
        generate_tier(tier)
        if tier == "small":
            # The 10K prefix backs the optional generated unit test.
            generate_small_prefix()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
