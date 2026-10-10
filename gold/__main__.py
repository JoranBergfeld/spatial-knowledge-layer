"""Gold for the layer's evaluation: template questions and what a correct turn would hold.

    python -I -S gold cobie --workbook handover.xlsx --ocr ocr/ --out gold.json
    python -I -S gold bundle --bundle bundles/2026-09-15 --out gold.json

This directory shares no code with the layer. It imports nothing but the standard
library, so it runs with site-packages switched off (-S), where the layer's
packages cannot even be found. That is on purpose: gold that used the layer's
tools would agree with them about their bugs.

`cobie` is the test bench: gold from the handover workbook alone. `bundle` is
for the demo venue, which has no handover, so its gold is read off the bundle.

The same input gives the same file, byte for byte. Nothing is drawn at random.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import from_bundle
import from_cobie
import questions
from world import PERSONAS, World

FORMAT = "spatial-gold@1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sizes(world: World) -> dict[str, Any]:
    return {
        "spaces": len(world.spaces),
        "asset_types": len(world.types),
        "assets": len(world.assets),
        "documents": len(world.documents),
        "tiers": world.tiers,
        "visible": {
            persona: {
                "spaces": len(world.visible[persona]["spaces"]),
                "asset_types": len(world.visible[persona]["types"]),
                "assets": len(world.visible[persona]["assets"]),
                "documents": sum(
                    1 for d in world.documents.values() if world.view(persona).reads(d)
                ),
            }
            for persona in PERSONAS
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gold", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cobie = commands.add_parser("cobie", help="gold from a COBie handover workbook")
    cobie.add_argument("--workbook", type=Path, required=True)
    cobie.add_argument(
        "--ocr", type=Path, help="the OCR'd text. A document with no text there is left out"
    )
    bundle = commands.add_parser("bundle", help="gold read off a snapshot bundle")
    bundle.add_argument("--bundle", type=Path, required=True)
    for command in (cobie, bundle):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument(
            "--per-template", type=int, default=6, help="questions per template, before pairing"
        )
    args = parser.parse_args(argv)

    if args.command == "cobie":
        world = from_cobie.read(args.workbook, args.ocr)
        source = {"kind": "cobie", "file": args.workbook.name, "sha256": _sha256(args.workbook)}
    else:
        world = from_bundle.read(args.bundle)
        manifest = args.bundle / "manifest.json"
        source = {"kind": "bundle", "file": manifest.name, "sha256": _sha256(manifest)}

    asked = questions.generate(world, args.per_template)
    gold = {
        "format": FORMAT,
        "source": source,
        "per_template": args.per_template,
        "personas": list(questions.ASKED_AS),
        "unmeasured": list(questions.UNMEASURED),
        "world": _sizes(world),
        "questions": asked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(gold, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    kinds: dict[str, int] = {}
    for question in asked:
        kinds[question["expected"]["kind"]] = kinds.get(question["expected"]["kind"], 0) + 1
    print(f"{len(asked)} questions to {args.out}: {json.dumps(kinds, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
