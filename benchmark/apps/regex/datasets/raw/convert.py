#!/usr/bin/env python3
"""Convert raw pcap payload to basis bit-plane .npz format.

Reads TCP/UDP payload bytes from a pcap file (or raw binary payload file),
transposes to 8 basis bit-planes (Parabix byte decomposition), and saves
as .npz for efficient loading during benchmarks.

Output format:
  - b0..b7: basis bit-planes (uint8 arrays, packed bits)
  - payload_length: number of bytes in original payload

Usage:
    python convert.py <input> <output.npz> [--max-bytes N]

Input can be:
  - .pcap file (extracts TCP/UDP payload)
  - .bin file (raw bytes)
"""

from __future__ import annotations

import struct
import sys
import os

import numpy as np


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


def bytes_to_basis_bits(payload: bytes) -> dict[str, np.ndarray]:
    """Transpose byte array to 8 basis bit-planes.

    Each bit-plane b[i] has bit i of every byte, packed into uint8 arrays.
    Bit j of byte k is stored at position k in the bit-plane.
    """
    n = len(payload)
    arr = np.frombuffer(payload, dtype=np.uint8)

    planes = {}
    for bit in range(8):
        # Extract bit `bit` from each byte
        plane = ((arr >> bit) & 1).astype(np.uint8)
        # Pack bits into bytes for compact storage
        planes[f"b{bit}"] = np.packbits(plane, bitorder='little')

    return planes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert pcap/binary to .npz basis bits")
    parser.add_argument("input", help="Input file (.pcap or .bin)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output .npz file (positional)")
    parser.add_argument("--npz", dest="npz_output", default=None,
                        help="Output .npz file (alternative to positional)")
    parser.add_argument("--max-bytes", type=int, default=None,
                        help="Maximum bytes to extract")
    parser.add_argument("--source", default=None,
                        help="Source description (stored in metadata)")
    args = parser.parse_args()

    # Resolve output path: --npz takes priority over positional
    out_path = args.npz_output or args.output
    if out_path is None:
        parser.error("output path required (positional or --npz)")

    # Load payload
    if args.input.endswith('.pcap'):
        print(f"Extracting payload from pcap: {args.input}")
        payload = extract_pcap_payload(args.input, args.max_bytes)
    else:
        print(f"Reading raw binary: {args.input}")
        with open(args.input, 'rb') as f:
            payload = f.read()
        if args.max_bytes:
            payload = payload[:args.max_bytes]

    print(f"Payload size: {len(payload)} bytes")

    # Convert to basis bits
    planes = bytes_to_basis_bits(payload)

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, payload_length=len(payload), **planes)
    file_size = os.path.getsize(out_path)
    print(f"Saved to {out_path} ({file_size} bytes compressed)")


if __name__ == "__main__":
    main()
