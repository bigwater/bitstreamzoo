"""Auto-discover benchmark domains from the filesystem."""

import os


def discover_domains(apps_dir=None):
    """Scan benchmark/apps/ for domains (dirs with src/*.bs files).

    Returns sorted list of domain names.
    """
    if apps_dir is None:
        apps_dir = os.path.join(os.path.dirname(__file__), "apps")
    domains = []
    for d in os.listdir(apps_dir):
        src_dir = os.path.join(apps_dir, d, "src")
        if not os.path.isdir(src_dir):
            continue
        if any(f.endswith(".bs") for f in os.listdir(src_dir)):
            domains.append(d)
    return sorted(domains)
