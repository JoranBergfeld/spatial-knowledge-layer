"""Fetching a handover's pinned files, once, by hand. The bench's version of the Overpass pull.

Each file is downloaded next to where it belongs, hashed, and only moved into
place when the hash is the pinned one. A file already there with the right hash
is left alone, so running it again costs nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from spatial_ingest.handover.definition import Definition, DefinitionError, sha256_of

_USER_AGENT = "spatial-knowledge-layer-ingest (github.com/JoranBergfeld/spatial-knowledge-layer)"


def fetch_handover(definition: Definition, directory: Path) -> dict[str, str]:
    """Every pinned file with a URL, into ``directory``. Returns file name to what happened."""
    directory.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}
    for pinned in definition.pinned():
        path = directory / pinned.file
        if path.is_file() and sha256_of(path) == pinned.sha256:
            done[pinned.file] = "already there"
            continue
        if pinned.url is None:
            raise DefinitionError(f"{pinned.file} has no URL to fetch it from")
        partial = path.with_name(path.name + ".partial")
        digest = hashlib.sha256()
        try:
            with httpx.stream(
                "GET",
                pinned.url,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_bytes():
                        digest.update(block)
                        handle.write(block)
        except httpx.HTTPError as error:
            partial.unlink(missing_ok=True)
            raise DefinitionError(f"could not fetch {pinned.url}: {error}") from error
        if digest.hexdigest() != pinned.sha256:
            partial.unlink()
            raise DefinitionError(
                f"{pinned.url} is not the file that was pinned "
                f"({digest.hexdigest()} is not {pinned.sha256})"
            )
        partial.replace(path)
        done[pinned.file] = "fetched"
    return done
