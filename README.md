# spatial-knowledge-layer

A retrieval system that answers questions about a physical place: its structure, the
assets installed in it, and the documents anchored to both. The demo venue is KIT
Campus Süd, from OpenStreetMap.

Ingestion runs here: `ingest build` turns a pinned OpenStreetMap pull into a snapshot
bundle holding the venue's structure, `ingest load` writes a bundle into Postgres and
checks it, and `ingest promote` makes a loaded snapshot current. The service answers
one kind of question so far, "where is room X". The seed inventory and documents are
later issues.

The spec and the glossary live in the site's repository
([spec](https://github.com/JoranBergfeld/personal-website/issues/73), `CONTEXT.md`).
Terms here are used as written there.

## Layout

```
packages/core     record models, the stores' DDL and the clearance rules, shared with the service
packages/ingest   the ingestion CLI, its own package and its own image
packages/service  the turn service: FastAPI on Microsoft Agent Framework, reading the stores
```

The service holds no pipeline code and imports nothing from `packages/ingest`.

## Build a snapshot

```
uv sync
uv run ingest build --date 2026-09-15
```

That pulls a dated Overpass query over the Campus Süd bounding box and writes
`bundles/2026-09-15/`:

```
pull/overpass.json      the response, as received
pull/query.overpassql   the query text
pull/source.json        where the pull came from
osm/spaces.jsonl        the containment tree, one space per line
osm/spans.jsonl         spaces covering more floors than the one they hang from
osm/edges.jsonl         the circulation edges: one way through a door, with its clearance
osm/visible_spaces.jsonl  what exists for each persona: visitor, staff, maintenance
manifest.json           input hashes, what produced each part, counts, the sourced-clearance share
```

Run it again and it reuses the stored pull and writes the same bytes. `--refetch`
replaces the pull. If the stored pull no longer matches the hash it was pinned with,
the build stops.

Overpass rate limits after a few heavy queries. The fallback is a dated regional extract
(Geofabrik has them), which the build cuts down to the same box and pins the same way:

```
uv run ingest build --date 2026-09-15 --extract karlsruhe-regbez-260915.osm.pbf
```

`bundles/` is gitignored. The pull is private and never committed.

## Load and promote

The layer has a Postgres database of its own on the site's instance. Ingestion owns its
DDL (`packages/core/src/spatial_core/ddl/`) and applies it on every run; nothing else
migrates it. Create the database once, owned by ingestion's login, which also needs
`CREATEROLE` to make the service's role:

```
CREATE ROLE ingest LOGIN CREATEROLE PASSWORD '...';
CREATE DATABASE spatial OWNER ingest;
```

Then, with `SPATIAL_DATABASE_URL` (or `--database-url`) naming that login:

```
uv run ingest load --date 2026-09-15      # write bundles/2026-09-15 and check it
uv run ingest promote --date 2026-09-15   # make it current
uv run ingest promote --date 2026-09-08   # roll back: the same command, the previous date
```

Load writes a snapshot as three schemas (ADR 0001 in the site's repository):

```
snap_20260915_osm    the bundle's osm/ half: sourced from OSM or derived by a rule (ODbL)
snap_20260915_seed   the bundle's seed/ half: generated data (empty until the seed inventory)
snap_20260915        views joining the two, which is all the service reads
```

It first checks each part against the sha256 in the manifest and parses every record,
then writes, then checks what it wrote:

- the tree has exactly one root
- every space has one parent, which is a space of the snapshot
- every ancestor path is its parent's path plus its parent
- every span names spaces of the snapshot
- every edge names spaces of the snapshot
- every visible space is a space of the snapshot, and its parent is visible too
- the visible sets nest: visitor within staff within maintenance
- the OSM schema holds no seed rows
- each table holds as many rows as the manifest says

A snapshot that fails any of them is recorded as failed in `control.snapshots` and
`promote` refuses it. Its schemas stay, for looking at. Loading the same bundle again
rebuilds the schemas from scratch, so it yields the same stores.

Load never moves the current-snapshot pointer (`control.current_snapshot`, one row), and
it refuses to reload the current or the previous snapshot, since conversations may be
held on either. Promote is one update of that row: the snapshot it names becomes current
and the one it replaces becomes previous. The next load drops every snapshot older than
both; snapshots loaded but not yet promoted stay until they are older than both too.

The service gets a login that is a member of `spatial_reader`. That role can read
`control` and the views schemas and nothing else; it cannot write anywhere. Ingestion's
login owns every object and is the only one that writes.

Each load and promote also writes its log to `bundles/<date>/runs/`. The logs are not
parts of the bundle, and the manifest does not list them.

## The rules

The clearance and visibility rules live in `packages/core/src/spatial_core/clearance.py`,
since the service imports them too. There are no per-element overrides: a rule that
gets a room wrong is fixed by changing the rule.

Every derived value in a bundle names the rule that made it. The rules are plain
functions under `packages/ingest/src/spatial_ingest/rules/`, each with its own tests.

| Rule | What it does |
| --- | --- |
| `site-from-venue-definition@1` | The site is the union of the venue's part ways. Its name comes from the venue definition, because the ways carry none. |
| `containment-by-polygon@1` | A space belongs to the site when its representative point lies in a part polygon, and to the smallest building polygon holding that point. |
| `tier-from-osm-tags@1` | `building=*` is a building. `indoor=room`, `corridor` and `area` sit at the room tier. |
| `floor-from-level-tags@1` | A building gets one floor per level its rooms name. Nobody maps floors as elements on this campus. |
| `level-parse@1` | Reads `level`: `;` lists, `a-b` ranges, half levels. A space hangs from its lowest floor and spans the rest. |
| `room-label@1` | The sourced name (one the name filter left untouched), or else the building's number (or name) plus the room number: "50.34 room 107". No rule, no label, and the space is marked unaddressable. |
| `edges-from-doors@1` | A node tagged `door` or `entrance` joins the room-tier spaces whose outline holds it, one edge each way. On one outline only, it opens from outside. |
| `clearance-from-access@1` | `yes`, `permissive`, `customers` are visitor. `private`, `no`, `delivery`, `permit`, `emergency` are staff. Any other value fails closed to maintenance. The raw tag is kept beside it. |
| `clearance-authored-default@1` | Untagged corridors and areas, and untagged rooms of a public function (toilets, auditorium, classroom, seminar room, hall, library, stairs, elevator, entrance), are visitor. Any other untagged room is staff. Flagged as an authored default, and never over a tag. Doors and spaces above the room tier get no default. |
| `edge-clearance@1` | An edge takes the stricter of its door's tag and the clearance of the space it enters. An untagged door constrains nothing. |
| `visible-by-edges@1` | A space is visible to a persona when an edge it clears enters the space, or, with no edge entering, by the space's own clearance. Its ancestors are visible with it. |
| `strip-person-names@1` | Removes academic titles with the name after them, and the capitalised words after Büro, Sekretariat, Labor or AG. Runs over names and descriptions. Rooms named after someone are left alone. |

Things that are rough or just not done yet:

- The name filter eats any capitalised noun after its four keywords, so
  "Sekretariat Geschäftsführung" loses a word. One hit on the real campus so far.
- Identical duplicate rooms are not deduplicated yet.
- `repeat_on` is ignored. A stairwell copied onto five levels is one space on its `level`.
- The site's name is typed into the venue definition. OSM has no name on the part ways.
- A room with no readable level hangs from its building. The floor is not guessed from
  the room number.
- A room inside the site but in no building polygon hangs from the site and cannot be
  addressed by number.
- A building with no rooms and no `access` tag is entered by no edge, so no persona sees it.
  Entrances on a building outline make edges only where a room's outline holds them too.
- A door node shared by spaces on different levels joins them. The door's `level` is not
  checked against theirs.

## The service

One operation, `POST /turn`. It takes a persona, a snapshot (none means the current
one), the conversation's prior state and a question, and returns a turn:

```
{"persona": "visitor", "snapshot": null, "state": null, "question": "Where is 50.34 room 107?"}
```

```
uv run spatial-service
```

It reads its configuration from the environment:

```
SPATIAL_SERVICE_DATABASE_URL     a login in spatial_reader; the service refuses one that can write
AZURE_OPENAI_ENDPOINT            the Azure OpenAI resource
AZURE_OPENAI_API_KEY             its key
AZURE_OPENAI_API_VERSION         optional, 2024-10-21 by default
SPATIAL_PLANNING_DEPLOYMENT      the deployment that plans
SPATIAL_COMPOSITION_DEPLOYMENT   the deployment that composes, which must be a different one
SPATIAL_LLM_TIMEOUT_SECONDS      optional, 30 by default
```

A turn is three stages, plus a fourth only when the plan cannot run. A turn makes at
most three model calls however many steps run (plan, recover, compose), and the host
enforces that cap rather than trusting the flow to keep it:

- **Plan.** The planning deployment returns a plan (`packages/service/src/spatial_service/plan.py`):
  the question's shape, the place mentions in it, `within` constraints between them,
  and a flat ordered list of typed tool steps. A step may work within the ids an
  earlier step produced. The plan is checked in code before any tool runs. Persona and
  snapshot are not arguments a plan can carry: the host binds them.
- **Execute.** The steps run in order with no model call between them. The one tool,
  `resolve_place`, only ever sees the persona's visible set in that snapshot. It tries
  an exact label, then a building and room number parsed the way `room-label@1` builds
  labels ("50.34 room 107", "50.34/107", "room 107 in 50.34"), then a number alone,
  then the words of a name. Spaces without a label are never answers.
- **Recover.** Only on a typed trigger (`packages/service/src/spatial_service/recovery.py`),
  the planning deployment is handed its plan back with the trigger and its reasons, and
  writes a second. That one is checked and run exactly like the first. There is no
  third: a second trigger ends the turn in the clarification still pending, if one is,
  and otherwise in a `plan-rejected` abstention.
- **Compose.** The composing deployment is told the facts by reference (`{f1}`, `{f2}`)
  with their tiers, never their labels or ids. It returns segments that cite them; the
  host fills each placeholder from its fact. Output with a digit anywhere outside a
  placeholder is rejected whole.

The triggers, and nothing else, buy a recovery call:

| Trigger | When |
| --- | --- |
| `plan-invalid` | The plan does not parse, or breaks a rule: the step cap, a repeated id, a step feeding nothing. |
| `unbound-reference` | A reference binds to nothing: a step resolving a mention the plan lacks, or reading a step that is not earlier or whose output it cannot take. It wins over `plan-invalid`, since a dangling reference breaks other rules in its wake. |
| `empty-retrieval` | Document retrieval returned nothing. Nothing retrieves documents yet, so nothing raises it so far. |
| `malformed-arguments` | A tool rejected its arguments, such as a mention with nothing to look up once normalised. |

Outcomes that are answers never buy a call: no visible place matching (which is also
what a place beyond clearance looks like), several matching, a question of an
unsupported shape. Every step's arguments are checked before the first step runs. Were
they checked as each step came up, a bad later step would be reached only when the
earlier ones resolved, and the recovery call would tell what the persona can see.

**The step cap is four.** The one supported shape needs a step per place mention, and
the longest chain worth reading in one question is three deep ("room 107 on level 1 of
50.34"), plus one to spare. A plan over the cap is rejected, not truncated, since a
shortened plan answers a different question.

A turn is one of four kinds:

| Kind | When |
| --- | --- |
| `answer` | Every step resolved one place and the composition passed its checks. |
| `clarification` | A step matched several places, or the mention was a bare room number, which is never a label. The candidates are all visible to the persona, at most six. The state carries them, and the next turn's planner sees their labels. |
| `abstention` | `no-such-place` when nothing the persona can see matches. A place beyond clearance and a place that does not exist give the same abstention, and the same trace but for the words asked. Also `unsupported-question`, `plan-rejected` and `composition-rejected`. |
| `unavailable` | A model could not be reached. HTTP 503, no answer, and the prior state back unchanged. |

Every turn carries its trace, the new conversation state, each model call's token usage
with the deployment that served it, and `recovery`: `not-needed`, `repaired` or
`unrepaired`, so the repair rate is a count over turns. A second plan that calls the
question unsupported runs nothing, so it counts as `unrepaired`. The trace holds the shape, the
step and call caps, what was resolved and how, the provenance of every label in the
answer, and any authored default clearance among them, and under `plans` each version
of the plan: as the model wrote it, its verdict and reasons, each step it ran with its
result ids, the trigger it raised, and for the second version its changes against the
first as JSON Pointer paths. Steps and mentions are matched by id, so one added or
dropped at the front shows as one change.

A conversation keeps the persona and snapshot it began with, and is refused once its
snapshot is neither current nor previous. State a client sends back is cut down to what
the persona can see before anything reads it.

Still to come: one question shape, one tool, and clarifications and abstentions are
worded by the host rather than composed.

## Develop

```
SPATIAL_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

The load and promote tests, and the service's store tests, need a Postgres superuser URL
in `SPATIAL_TEST_DATABASE_URL` and make databases of their own. Without it they skip,
except in CI. The turn tests run against a fixture snapshot in memory and a scripted
model, and need neither.

## Data

Venue structure is © OpenStreetMap contributors, under the
[ODbL](https://opendatacommons.org/licenses/odbl/). The `osm/` half of a bundle is the
derived database, and the rules above are the method.
