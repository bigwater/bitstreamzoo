"""Tier configuration for standardized dataset sizes.

3 tiers defined by total binary data size (input + output streams):
  Small:  ~50MB
  Medium: ~500MB
  Large:  ~5GB

bitlength = target_bytes * 8 / n_binary_streams
"""

# bitlength per domain per tier.
# Key: domain name (or circuit name for circuit_sim).
# Values: (small, medium, large) or None if tier is skipped.
TIER_BITLENGTH = {
    # domain: (n_streams, small, medium, large)
    "epistasis":      (3,   133_000_000,  1_330_000_000,  13_300_000_000),
    "shift_and":      (5,    80_000_000,    800_000_000,   8_000_000_000),
    "mison":          (12,   33_300_000,    333_000_000,   3_330_000_000),
    "bitweaving":     (10,   40_000_000,    400_000_000,   4_000_000_000),
    # Montgomery mul: a[K]+b[K]+r[K] = 3*K = 36 binary streams. n[K] and np[K]
    # are broadcast constants stored inline (negligible), so they are excluded
    # from the size estimate (cf. GEMV weights). K=12 for Kyber's prime q=3329;
    # tiers land at ~30/300/3000 MB (documented in the app readme).
    "montgomery_mul": (36,    6_666_667,     66_666_667,     666_666_667),
    "aes_sbox":       (16,   25_000_000,    250_000_000,   2_500_000_000),
    "edit_distance":  (36,   11_100_000,    111_000_000,   1_110_000_000),
    "simon":          (64,    6_250_000,     62_500_000,     625_000_000),
    # GEMV: n_streams = N*K (input) + L*B (output), weights are broadcast constants
    "gemv_small_k2":  (160,    2_500_000,   25_000_000,  250_000_000),
    "gemv_medium_k4": (1_152,    347_222,    3_472_222,   34_722_222),
    "gemv_large_k8":  (2_496,    160_256,    1_602_564,   16_025_641),
    # BNN large: exceeds the per-test time budget; not generated.
    "bnn":            (132,   3_030_303,     30_303_030,     None),
    "trivium":        (224,   1_790_000,     17_900_000,     179_000_000),
    # Circuit sim: each circuit has different pin counts.
    # n_streams = n_inputs + n_outputs (the streams written to .bsdata).
    # bitlengths match the committed .bsdata; do not change without regenerating.
    "c17":            (7,    57_000_000,    571_000_000,   5_710_000_000),
    # adder4 (local 4-bit adder, not ISCAS'85): round 20M/200M/2B
    # bitlengths, ~35/350/3500 MB payloads.
    "adder4":         (14,   20_000_000,    200_000_000,   2_000_000_000),
    "c432":           (43,    9_300_000,     93_000_000,     930_000_000),
    "c499":           (73,    5_500_000,     54_800_000,     548_000_000),
    "c880":           (86,    4_650_000,     46_500_000,     465_000_000),
    "c1355":          (73,    5_500_000,     54_800_000,     548_000_000),
    "c1908":          (58,    6_900_000,     69_000_000,     690_000_000),
    # c2670: 157 inputs + 64 outputs = 221, but output G2592 (= ~G2524) is a
    # structurally constant-0 stream, stored inline (not as a binary section),
    # so the .bsdata payload holds 220 binary streams.
    "c2670":          (220,   1_850_000,     18_500_000,     185_000_000),
    "c3540":          (72,    5_600_000,     55_600_000,     556_000_000),
    "c5315":          (301,   1_390_000,     13_900_000,     139_000_000),
    "c6288":          (64,    6_250_000,     62_500_000,     625_000_000),
    "c7552":          (315,   1_270_000,     12_700_000,     127_000_000),
    # BitFunnel: K0=3 + K1=3 + K2=2 = 8 input + 1 matches output = 9 streams (N docs)
    "bitfunnel":      (9,   50_000_000,    500_000_000,   5_000_000_000),
    # Regex: 8 input streams + 1 any_match output = 9 streams (all tiers)
    "regex_small":    (9,   44_444_444,    444_444_444,   4_444_444_444),
    # regex_medium: large tier infeasible (0.6 GB/var × ~500 max_live = ~300 GB OOM)
    "regex_medium":   (9,   44_444_444,    444_444_444,   None),
    # regex_large: medium tier GPU OOM (2400 live vars × 55.6 MB = 133 GB > any GPU VRAM),
    # CPU-only at 74 min exec. Large tier OOM on CPU too (2400 × 555 MB = 1.3 TB).
    "regex_large":    (9,   44_444_444,    None,          None),
    # Polar code encoding: n_streams = 2*N (N input + N output)
    "polar_small":    (128,   3_125_000,     31_250_000,     312_500_000),
    "polar_medium":   (512,     781_250,      7_812_500,      78_125_000),
    "polar_large":    (2048,    195_312,      1_953_125,      19_531_250),
}


def get_tier_bitlength(domain: str, tier: str) -> int | None:
    """Get bitlength for a domain at a given tier.

    Returns None if the tier is not applicable for this domain.
    """
    if domain not in TIER_BITLENGTH:
        raise ValueError(f"Unknown domain: {domain}")
    _, small, medium, large = TIER_BITLENGTH[domain]
    if tier == "small":
        return small
    elif tier == "medium":
        return medium
    elif tier == "large":
        return large
    else:
        raise ValueError(f"Unknown tier: {tier}")


def estimate_size_bytes(domain: str, tier: str) -> int | None:
    """Estimate the .bsdata file size in bytes for a domain at a given tier."""
    bitlength = get_tier_bitlength(domain, tier)
    if bitlength is None:
        return None
    n_streams = TIER_BITLENGTH[domain][0]
    # Each stream: ceil(bitlength / 8) bytes in .bsdata
    bytes_per_stream = (bitlength + 7) // 8
    return n_streams * bytes_per_stream


# Backward-compat aliases for downstream generate_tests.py scripts
TIER_VECTORS = TIER_BITLENGTH
get_tier_vectors = get_tier_bitlength
