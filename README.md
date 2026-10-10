# spatial-knowledge-layer

A retrieval system that answers questions about a physical place: its structure, the
assets installed in it, and the documents anchored to both. The demo venue is KIT
Campus Süd, from OpenStreetMap.

Ingestion runs here: `ingest build` turns a pinned OpenStreetMap pull into a snapshot
bundle holding the venue's structure and its equipment, `ingest load` writes a bundle
into Postgres and checks it, and `ingest promote` makes a loaded snapshot current. The
service answers three kinds of question so far: where a place is, what is installed in
one, and which floor or building has the most of something. Documents are a later issue.

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
osm/asset_types.jsonl   the fire equipment types OSM maps (fire-assets-from-emergency@1)
osm/assets.jsonl        the fire equipment itself, sourced
osm/visible_assets.jsonl, osm/visible_asset_types.jsonl   which of those exist per persona
seed/asset_types.jsonl  the generated asset types
seed/assets.jsonl       the generated assets
seed/maintenance_tasks.jsonl  planned jobs per type, after COBie 2.4 Job
seed/service_events.jsonl     the latest service per asset, after COBie v3 Event
seed/visible_assets.jsonl, seed/visible_asset_types.jsonl
manifest.json           input hashes, what produced each part, counts, the sourced-clearance
                        share, the inventory's seed, fill rates and attribute catalogue
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

## The seed inventory

The demo venue gets equipment from a published, deterministic generator,
`packages/ingest/src/spatial_ingest/seed/generate.py`. Its one input besides the
structure is `seed/inventory.json`, which holds the fixed seed and is otherwise adapted
from the test bench's COBie handover: the type library, the bench's space categories with
their equipment mixes, its measured fill rates and install-date spread, and which room
types map to which category. The manifest records the file's sha256 and the seed, and
the same inputs give the same inventory, byte for byte. `--inventory <file>` builds from
another one.

Equipment mixes and type library adapted from the buildingSMART Medical-Dental Clinic
sample (CC BY 4.0). The file names the handover and its sha256.

Only the domain-neutral families are generated: lights, AC units, air terminals, doors,
valves, sanitary fittings and electrical panels. Each generated type is the Clinic's most
installed type with its Table 23 code, and stands for every Clinic type with that code.
The generator refuses a type outside those families, and a camera by name. What the
Clinic leaves blank stays blank, so most types have no expected life.

Every generated record is `seed`, every value on it is `inferred` with its basis, and it
is written only to the bundle's `seed/` half. The fire equipment OSM maps
(`emergency=fire_extinguisher`, `fire_hydrant`, ...) is read by a rule instead, stays
`sourced`, and goes to the `osm/` half.

| Rule | What it does |
| --- | --- |
| `room-category@1` | A room-tier space takes the Clinic space category of the first mapping whose tags it carries (`room=office` is an Office, `room=auditorium` a Classroom, `indoor=corridor` a Corridor). An unmapped room gets nothing. |
| `count-by-area@2` | Per item of the category's mix, the room's area over the item's square metres on the Clinic, rounded, at least its min. The min is 1 where the median Clinic space of the category holds one. A room without an outline counts with the category's typical area. |
| `layout-in-outline@1` | Floor-standing things on a grid inside the real outline, wall-mounted things along it, both 0.5 m in from the walls. A room without an outline gets its count and no placement. |
| `door-asset@1` | A door for each sourced door an edge goes through, in the mapped space it opens into. It is bound to the door; the edge stays the door's, and no asset is ever an edge. |
| `manufacturer-from-email@1` | The Clinic names a manufacturer by a contact address. The name is the label before the top-level domain (`sales@lithonia.com` is Lithonia). A placeholder address reads not recorded. |
| `fill-by-rate@1` | Serial number on 43% of a type's assets, installation date and warranty start on all of them: the shares measured on the Clinic's 5,071 components. The rest are gaps. |
| `install-date@2` | As long before the snapshot as a draw from the Clinic's spread of installation dates before its handover, and never before the building's sourced `start_date` where OSM has one. |
| `warranty-start@1` | The installation date. On the Clinic the two agree on every component. |
| `serial@1` | The type's model and eight hex digits. |
| `latest-service@2` | One event on 70% of assets: a start and a performer, no end and nothing about the outcome. It falls within one interval of the type's most frequent task before the snapshot, or within a year for a type with no task. The share and the performer are chosen here, not measured, since the Clinic has no service events. |
| `fire-assets-from-emergency@1` | A node tagged with a fire `emergency` value is a sourced asset of a type derived from that value, under Table 23 `23-65 70 17 11` (fire fighting terminal components, where the Clinic files its extinguisher cabinets), or `23-65 70` for a call point. |
| `asset-in-space-by-point@1` | A sourced asset sits in the smallest room-tier space covering it on its level, else the smallest building, else the site. Outside the site it is dropped. |
| `clearance-from-omniclass@1` | A type's clearance from its OmniClass Table 23 code, by the longest prefix in the published three-tier table. Only the numeric code is matched, never the title. It is always an authored default. An unlisted or missing code fails closed to maintenance, flagged as an unrecognised category. |
| `visible-assets@1` | An asset is visible when its type is cleared and its space is visible. A type is visible only through a visible asset. |

Every asset carries its ancestor path: its space's path and then its space.

The Clinic has no maintenance job for any of the generated types. The two jobs in the
file (the door's and the panelboard's) are authored, and the file says so.

The manifest counts the types whose code fell back (`inventory.unrecognised_categories`),
the way `clearance.unrecognised_access` does for spaces. It never blocks a load. The
attribute catalogue (per asset type, which attributes at least one asset carries) is in
the manifest and in the `attribute_catalogue` view.

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
snap_20260915_seed   the bundle's seed/ half: the generated inventory
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
- the OSM schema holds no seed rows, and the seed schema only seed rows
- the seed schema holds no edges
- every asset sits in exactly one space, and is of a type of the snapshot
- every asset's ancestor path is its space's path plus its space
- a door asset is bound to a door of the snapshot
- tasks name their type; events name their asset and task
- a visible asset's type is cleared and its space is visible, and a type is visible only
  through a visible asset
- the visible assets and the visible asset types nest, like the spaces
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

The clearance and visibility rules, for spaces and for assets, live in
`packages/core/src/spatial_core/clearance.py`, since the service imports them too. There are no per-element overrides: a rule that
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
SPATIAL_TREE_NODE_CAP            optional, 200 by default: how much of the tree the planner sees
OTEL_EXPORTER_OTLP_ENDPOINT      optional; where metrics are pushed. Unset, none are
```

### Metrics

The service says how it is used, not who uses it
(`packages/service/src/spatial_service/metrics.py`). Three counters, each a count of
turns, pushed over OTLP/HTTP to `OTEL_EXPORTER_OTLP_ENDPOINT` plus `/v1/metrics`:

```
spatial.turns            every turn
spatial.repair_calls     turns that spent a recovery call, repaired or not
spatial.coverage_gaps    turns that ended in a coverage-gap abstention
```

The repair-call rate is `spatial.repair_calls` over `spatial.turns`, worked out wherever
the metrics land. Every point has the same four labels: `shape`, `persona`, `turn_kind`
and `snapshot`. There is no visitor, conversation, space or asset id, and a test holds
the set to those four.

- `shape` is `unknown` when no plan got as far as running, which covers a model that
  could not be reached.
- `turn_kind` is the turn's kind, except that a coverage-gap abstention is
  `coverage-gap` here and not `abstention`. Two things produce it: a building or
  bigger with no recorded installations, and an attribute no asset in scope carries.
- A request the service refuses (409, 503 with no snapshot) is not a turn and is not
  counted.
- Headers, timeout and the export interval come from the usual `OTEL_` variables, 60
  seconds between pushes by default. `OTEL_SERVICE_NAME` overrides `spatial-service`.
- With no endpoint the OTel SDK is never loaded and recording does nothing.

A turn is three stages, plus a fourth only when the plan cannot run. A turn makes at
most three model calls however many steps run (plan, recover, compose), and the host
enforces that cap rather than trusting the flow to keep it:

- **Plan.** The planning deployment returns a plan (`packages/service/src/spatial_service/plan.py`):
  the question's shape, the place and asset-type mentions in it, `within` constraints
  between places, and a flat ordered list of typed tool steps. A step may work within
  the ids an earlier step produced. The plan is checked in code before any tool runs.
  Persona and snapshot are not arguments a plan can carry: the host binds them. The
  planner is shown the collapsed tree (below), not the venue.
- **Execute.** The steps run in order with no model call between them. Every tool only
  ever sees the persona's visible set in that snapshot. `resolve_place` tries an exact
  label, then a building and room number parsed the way `room-label@1` builds labels
  ("50.34 room 107", "50.34/107", "room 107 in 50.34"), then a number alone, then the
  words of a name. Spaces without a label are never answers. The asset tools are below.
- **Recover.** Only on a typed trigger (`packages/service/src/spatial_service/recovery.py`),
  the planning deployment is handed its plan back with the trigger and its reasons, and
  writes a second. That one is checked and run exactly like the first. There is no
  third: a second trigger ends the turn in the clarification still pending, if one is,
  and otherwise in a `plan-rejected` abstention.
- **Compose.** The composing deployment is told the facts by reference (`{f1}`, `{f2}`)
  with their kinds, never their labels, names, numbers or ids. It returns segments that
  cite them; the host fills each placeholder from its fact. Output with a digit anywhere
  outside a placeholder is rejected whole, and so is an answer that leaves out a fact
  it has to name.

The triggers, and nothing else, buy a recovery call:

| Trigger | When |
| --- | --- |
| `plan-invalid` | The plan does not parse, or breaks a rule: the step cap, a repeated id, a step feeding nothing. |
| `unbound-reference` | A reference binds to nothing: a step resolving a mention the plan lacks, or reading a step that is not earlier or whose output it cannot take. It wins over `plan-invalid`, since a dangling reference breaks other rules in its wake. |
| `empty-retrieval` | Document retrieval returned nothing. Nothing retrieves documents yet, so nothing raises it so far. |
| `malformed-arguments` | A tool rejected its arguments, such as a mention with nothing to look up once normalised, or a scale level the snapshot has no tier for. |

Outcomes that are answers never buy a call: no visible place or asset type matching
(which is also what one beyond clearance looks like), several places matching, nothing
installed in scope, a question of a shape the layer declines. Every step's arguments are checked before the first step runs. Were
they checked as each step came up, a bad later step would be reached only when the
earlier ones resolved, and the recovery call would tell what the persona can see.

**The step cap is six.** The longest place chain worth reading in one question is
three deep ("room 107 on level 1 of 50.34"), then a type, the assets and a count over
them. A plan over the cap is rejected, not truncated, since a shortened plan answers a
different question.

### Assets

Four question shapes run on four tools. `place-location` is "where is room X".
`asset-containment` is "what is installed in room X" and "where are the lights on floor
2". `asset-aggregation` is "which floor has the most lights". `attribute-filter` is
"which lights have no serial number" and "what falls due for service before June".

| Tool | What it does |
| --- | --- |
| `resolve_place` | One place a mention names. As above. |
| `resolve_asset_type` | The visible asset types a mention names: an exact name, else every word of the mention in a type's name ("lights" is every type of light). A set, so nobody is asked which. |
| `assets_in_scope` | Every visible asset under one scope node, or under the whole venue, of the given types if any, passing the given attribute filters if any. One step and one pass over ancestor paths, however many spaces lie under the scope. No call per space. |
| `aggregate` | Those assets counted per space of one tier under the scope. |

A scale level named in the question sets two things. The place it names is the scope
("the Informatik building"), and the tier it compares across is the grouping ("which
floor"). The tiers are the snapshot's own, so `aggregate` takes whatever the venue has.

Aggregation walks containment only. An asset counts under the spaces on its ancestor
path, so one in a stairwell covering three floors counts once, on the floor the
stairwell hangs from. The service never reads `spans` for a count.

Counts cover what the persona can see and nothing else. The tools are handed a venue
already cut to the visible set, so a hidden asset is not filtered out of a result, it
was never in it. Nothing counts what was left out, so no answer can mention it. Ask the
same question as visitor and as maintenance and the numbers differ, that is all.

A count is never one number over sourced and generated assets. Every tally keeps the
two apart, and there is no fact for their sum, so the composing model has nothing to
print it from. A generated count fills in as "3 generated" and a sourced one beside it
as "1 sourced". The host writes those words, not the model. An answer that names one
half of a split count without the other is rejected.

No recorded installations is not zero. A building, or anything bigger, with nothing
visible installed in it at all is a `coverage-gap` abstention, worded by the host, with
no composition call. That gap is only ever stated at building level or above. Ask about
a room or a floor with nothing in it and the layer looks at its building: if the whole
building is empty the gap is stated for the building, and if not the turn is a plain
`nothing-recorded` abstention that gives no reason. A room is too small a place to call
a gap in the venue's data, and it still is not a zero. In a count across floors, a
floor with nothing recorded is listed apart and never ranked, and the answer has to
name it. A floor with other things installed and none of the type asked for is a zero,
and is ranked.

An asset without a placement reads as `position not recorded` in the trace, and the
answer can say how many of the assets it found have none. A placement is never printed
as coordinates, anywhere.

A question that needs one step's result to choose the next ("what is in the room with
the most lights") is its own shape, `step-dependent`, and gets an abstention. The plan
schema could not run it anyway: nothing reads what `aggregate` produces.

### Attribute filters and service due

`assets_in_scope` takes up to three filters. A filter is an attribute and one of four
tests, and that is the whole vocabulary:

| Test | Passes when |
| --- | --- |
| `recorded` | the asset carries the attribute |
| `not-recorded` | it does not |
| `before`, `after` | the attribute is a date and falls before or after the `date` given |

There is no free predicate. A filter has three fields and the step has no other place
to put one, an attribute's name has to look like a name, and anything else does not
parse. That is a `plan-invalid` trigger.

Which attributes exist is the attribute catalogue's to say: per asset type, the
attributes at least one asset carries. Nothing declares it. The service reads it off
the assets, cut to what the persona can see, and the planner is shown the names under
`venue.attributes`. A filter on anything else is a `malformed-arguments` trigger that
lists the names, checked before any step runs. An attribute only hidden assets carry is
not in a visitor's catalogue, so it is rejected in the same words as one that does not
exist. `before` and `after` on an attribute that is not a date are rejected the same way.

Two attributes are not stored on any asset and are in the catalogue anyway:

- `service_event` is the start of the asset's latest service event.
- `service_due` is that, or the installation date where there is no event, plus one
  year. It is worked out when a question asks and stored nowhere. The year is a default
  interval, the same for every asset type. It is nobody's standard, and the trace
  flags it: `default_interval`, `interval_years` and the rule `service-due@1`, with each
  asset's due date and the date it was counted from.

An asset that does not carry an attribute is not recorded for it, and that is all the
layer says. It does not pass or fail `before` or `after`. It is counted apart, and the
answer has to name that count, so "one is due" cannot pass for "the rest are not". The
host writes the words: the condition fills in as "service due before 2026-01-01" and
the gap as "no serial number recorded", and the model only places them. An answer that
says "never" is thrown out, which is a blunt check and the only one on the model's own
words.

The trace carries the fill rate of every attribute filtered on, under `assets.filters`:
how many of the assets asked about carry it, out of how many, counting only what the
persona can see. "Asked about" is the scope and the types in the question, so the rate
for "lights in the Audimax" is over the lights and not over everything in the room.

A fill rate of zero is a `coverage-gap` abstention with `gap` set to
`attribute-not-recorded`. The host names the attribute and no model is called. This one
is stated at any tier, since it is about an attribute and not a place. If there are no
assets of the type in scope at all there is no rate, and the answer is a count of zero
as before.

A filter also works under `aggregate` ("which building has the most lights with no
service event"), in an `asset-aggregation` plan.

### The collapsed tree

The planner is shown the top of the containment tree: labels and tiers of the spaces
the persona can see, cut by code at the deepest level where everything down to it still
fits under `SPATIAL_TREE_NODE_CAP` spaces. What lies below the cut is given as counts
per tier. The model never expands it. A plan names a scope in words and the tools look
under it.

The depth is measured every turn and recorded in the trace under `tree`: the depth of
the cut, the depth of the whole visible tree, and how many spaces each has. On the test
fixture (12 spaces for a visitor) the default cap shows all four levels and a cap of 8
cuts it at the floors. It has not been measured on the real campus yet, since there is
no bundle to hand here. 200 is a guess until then. With a few hundred buildings it will
most likely cut at the buildings or above.

A turn is one of four kinds:

| Kind | When |
| --- | --- |
| `answer` | Every step found what it needed and the composition passed its checks. |
| `clarification` | A step matched several places, or the mention was a bare room number, which is never a label. The candidates are all visible to the persona, at most six. The state carries them, and the next turn's planner sees their labels. |
| `abstention` | `no-such-place` when nothing the persona can see matches. A place beyond clearance and a place that does not exist give the same abstention, and the same trace but for the words asked. `no-such-asset-type` is the same for a type. Also `nothing-at-scale-level`, `step-dependent-question`, `unsupported-question`, `plan-rejected` and `composition-rejected`. `nothing-recorded` is a room or floor with nothing in it, inside a building that has installations: no reason given. And `coverage-gap`, with the turn's `gap` saying which: `no-recorded-installations`, when nothing the persona can see is installed in a building or anything bigger, or `attribute-not-recorded`, when no asset asked about carries the attribute filtered on. A gap in the data, not a count of zero. Nothing else gives a reason. |
| `unavailable` | A model could not be reached. HTTP 503, no answer, and the prior state back unchanged. |

Every turn carries its trace, the new conversation state, each model call's token usage
with the deployment that served it, and `recovery`: `not-needed`, `repaired` or
`unrepaired`, so the repair rate is a count over turns. A second plan that calls the
question unsupported runs nothing, so it counts as `unrepaired`. The trace holds the shape, the
step and call caps, the tree's measure, what was resolved and how, the provenance of
every label in the answer, and any authored default clearance among them. For an asset
question it holds the types asked for with the rule that cleared them, and under
`assets` the scope, the grouping, the rows as numbers, and every asset touched with
both provenance fields: record provenance, and the value provenance of its name and
placement with the rule or basis. Assets are listed one by one up to 200. Past that
they are still all accounted for, by provenance class, and the steps carry every id.
Under `plans` it holds each version
of the plan: as the model wrote it, its verdict and reasons, each step it ran with its
result ids, the trigger it raised, and for the second version its changes against the
first as JSON Pointer paths. Steps and mentions are matched by id, so one added or
dropped at the front shows as one change.

A conversation keeps the persona and snapshot it began with, and is refused once its
snapshot is neither current nor previous. State a client sends back is cut down to what
the persona can see before anything reads it. It holds the place and the asset types the
last answer was about.

Rough or not done yet on the asset side:

- Asset types match on the words of their name, with a trailing "s" folded off. "AC
  units" finds "AC Unit Type 1". "Air conditioners" finds nothing, and neither does any
  synonym. The planner is not shown the type names either.
- A space with no label cannot be named, so it cannot win "which has the most". Its
  count is in the trace and the answer gets one row for everything in unlabelled spaces.
- An answer spells out twelve rows and counts the rest.
- A campus-wide question puts every asset id in the trace's step. That is a lot of ids.
- Nothing uses an asset's placement yet beyond saying whether it is recorded.
- A filter compares against a date the plan carries. The planner is not told today's
  date, so "what is overdue" only works if the question names a day.
- A filtered answer counts assets by type or by space. It does not list them one by one
  with their due dates. Those are in the trace.
- The catalogue is kept per type but checked flat: a filter on an attribute that some
  other visible type carries passes the check and comes out as a fill rate of zero.
- Filters are not kept in the conversation state.
- There is no test for a value ("made by X"), only for presence and for dates.
- Clarifications, abstentions and coverage gaps are worded by the host, not composed.
- The plan schema is a plain union of step types, because strict structured output
  refuses `oneOf`. Whether the real deployment takes it has not been tried.

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
