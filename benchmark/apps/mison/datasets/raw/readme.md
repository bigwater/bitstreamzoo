# Mison JSON Structural Index -- Real-Life Data Provenance

## Source

GitHub Archive (GH Archive) JSON event data.

## Origin

- **GH Archive**: https://www.gharchive.org/
- Records every public GitHub event (pushes, PRs, issues, etc.) as
  newline-delimited JSON (NDJSON), one file per hour.
- Typical hourly file: 50-200 MB compressed, 200-800 MB uncompressed.

## Format

Newline-delimited JSON (`.json.gz`). Each line is a complete JSON object
representing one GitHub event with nested structures, string values
containing escaped characters, arrays, and various structural characters.

## Download

```bash
# GH Archive: hourly event files (newline-delimited JSON, gzipped)
wget "https://data.gharchive.org/2024-01-01-0.json.gz"    # ~72 MB compressed, ~500 MB uncompressed
wget "https://data.gharchive.org/2024-01-01-1.json.gz"
# Full day (24 files): for h in $(seq 0 23); do wget "https://data.gharchive.org/2024-01-01-$h.json.gz"; done
gunzip 2024-01-01-0.json.gz

# Alternative: Wikimedia recent-changes stream (JSON events)
curl -s "https://stream.wikimedia.org/v2/stream/recentchange" | head -n 10000 > wikimedia_events.json

# Alternative: NOAA weather API (JSON responses)
curl "https://api.weather.gov/stations/KORD/observations?limit=500" -o noaa_weather.json
```

## Conversion

`convert.py` reads JSON text files and decomposes each character into
8 ASCII basis bitstreams (b0..b7, Parabix-style bit planes). The Mison
pipeline then classifies structural characters ({, }, [, ], :, ,),
detects escapes, computes string masks, and filters structural indices.

## Converted datasets

Downloaded from GH Archive. Derived by `datasets/make_data.py`
(`text_to_basis_npz()`, driven by `python build.py mison`),
stored as 8 packed numpy basis bitstreams (b0..b7, Parabix bit planes).

- **`small/gharchive_1m.npz`** — First 1M characters from
  `https://data.gharchive.org/2024-01-01-0.json.gz`.
- **`medium/gharchive_10m.npz`** — First 10M characters from same file.
- **`large/gharchive_100m.npz`** — First 100M characters from same file.

All three are generated locally (`python build.py mison`) and are not
committed: the GH Archive dump contains real usernames and commit email
addresses, which the suite does not redistribute.

Original download: `https://data.gharchive.org/2024-01-01-0.json.gz` (~72 MB compressed).

## Alternative sources

- Twitter API stream archives (JSON)
- Wikimedia recent-changes stream (Server-Sent Events with JSON payloads)
- Common Crawl WAT/WET metadata (JSON format)
