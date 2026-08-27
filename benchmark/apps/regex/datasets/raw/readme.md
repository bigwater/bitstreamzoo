# Data Provenance

## WRCCDC 2012 Network Capture

- **Source**: Western Regional Collegiate Cyber Defense Competition, 2012
- **URL**: https://archive.wrccdc.org/pcaps/2012/wrccdc2012.pcap.gz
  (fetched by `python build.py regex`)
- **Format**: PCAP (libpcap), 28 MB
- **Contents**: 37,270 packets of TCP/UDP network traffic
- **License**: Publicly available for research
- **Local sha256**:
  - `wrccdc2012.pcap.gz`: `5a82f748df030ec274aa5e79d81f575fa000450405dd4238c4b0c24caad4f1fe`
  - `wrccdc2012.pcap` (decompressed): `cfb442509ba251dc960dceece05951fdbf5d1abe3fe60aae56067d1608e630de`

## Snort Patterns

- **Source**: AutomataZoo benchmark suite (Wadden et al., IISWC 2018)
- **URL**: https://github.com/hplp/AutomataZoo (canonical fork)
- **Original file**: `Snort/benchmarks/automata/snort.regex` (2,396 patterns)
- **Local copy**: `snort.regex` in this directory
- **Contents**: PCRE patterns from Snort 2.9.7.0 IDS rules
- **License**: GPL (Snort rules)
- **sha256**: `15d11c3e648b6089486a3053e7abcce1513fc4a10ef2cbf6433315f5b202908b`
  — verified bit-identical to
  `https://raw.githubusercontent.com/hplp/AutomataZoo/master/Snort/benchmarks/automata/snort.regex`
  on 2026-05-14.

### Curation: 2,396 → 2,312 patterns

The pattern pipeline is:

```
snort.regex (2,396 PCRE)
  -> src/curate_patterns.py     # filter + sort by op count ascending
  -> raw/pattern_manifest.json  # canonical pattern + tier definitions
  -> src/compile_all.py         # icgrep --ShowPablo -> pablo_to_bs.py
  -> src/regex_{small,medium,large}.bs
```

1. **icgrep / pablo_to_bs compilability**: of the 2,396 AutomataZoo
   `snort.regex` patterns, **2,312 compile** successfully through
   `icgrep --ShowPablo` → Pablo IR → `pablo_to_bs.py` 3-address `.bs`;
   **84 fail** (51 counted repetitions `{n}`/`{n,m}`, which icgrep
   compiles to unsupported `IndexedAdvance`; 27 non-greedy quantifiers).

2. **Sorting and tiers**: the 2,312 compilable patterns are sorted by
   operation count (ascending) and recorded in
   `raw/pattern_manifest.json` (`total_patterns`, `compiled_patterns`,
   `failed_patterns`, `tier_sizes`, and per-pattern `patterns`/`failed`
   lists). Tiers are **nested prefixes** of this sorted list, with sizes
   from `TIER_SIZES` in `src/compile_all.py`:

   | Tier | Patterns | Source of size |
   |------|----------|----------------|
   | `regex_small`  | 50    | `TIER_SIZES["regex_small"] = 50`     |
   | `regex_medium` | 500   | `TIER_SIZES["regex_medium"] = 500`   |
   | `regex_large`  | 2,312 | `TIER_SIZES["regex_large"] = None` (all compiled) |

   Per-pattern and per-tier operation counts are recorded in
   `pattern_manifest.json` (the single source of truth for tier membership).

## Conversion

`convert.py` extracts TCP/UDP payload bytes from the raw pcap file
and transposes them into 8 basis bit-planes stored as numpy `.npz` files.

```bash
python convert.py wrccdc2012.pcap out.npz [--max-bytes N]
```

converts one payload file to one basis-bitplane `.npz`. The tiered
datasets below are produced by `datasets/make_data.py` (driven by
`python build.py regex`):
- **`small/wrccdc2012_10k.npz`** and **`small/wrccdc2012_1m.npz`**
- **`medium/wrccdc2012_10m.npz`**
- **`large/wrccdc2012_100m.npz`**
