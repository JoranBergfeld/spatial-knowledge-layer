# spatial-knowledge-layer

A retrieval system that answers questions about a physical place: its structure, the
assets installed in it, and the documents anchored to both. The demo venue is KIT
Campus Süd, from OpenStreetMap.

Ingestion runs here: `ingest build` turns a pinned OpenStreetMap pull and a document
corpus into a snapshot bundle holding the venue's structure, its equipment and its
prose, `ingest load` writes a bundle into Postgres and a search index and checks both,
and `ingest promote` makes a loaded snapshot current. The service answers where a
place is, what is installed in one, which floor or building has the most of something,
and what a manual or a place's page says, with the passage cited. It also takes
follow-ups.

The spec and the glossary live in the site's repository
([spec](https://github.com/JoranBergfeld/personal-website/issues/73), `CONTEXT.md`).
Terms here are used as written there.

## Layout

```
packages/core     record models, the stores' DDL and the clearance rules, shared with the service
packages/ingest   the ingestion CLI, its own package and its own image
packages/service  the turn service: FastAPI on Microsoft Agent Framework, reading the stores
packages/bench    the evaluation harness: question sets, a scripted model, the leak checks,
                  evaluation runs
gold/             gold answers from a handover, by scripts that import none of the above
questions/        the hand-written questions. 52, drafted by a model
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
osm/edges.jsonl         the circulation edges: one way through a door, with its clearance,
                        where the door is and its level. Plus the snaps of doorless rooms
osm/visible_spaces.jsonl  what exists for each persona: visitor, staff, maintenance
osm/asset_types.jsonl   the fire equipment types OSM maps (fire-assets-from-emergency@1)
osm/assets.jsonl        the fire equipment itself, sourced
osm/visible_assets.jsonl, osm/visible_asset_types.jsonl   which of those exist per persona
seed/asset_types.jsonl  the generated asset types
seed/assets.jsonl       the generated assets
seed/maintenance_tasks.jsonl  planned jobs per type, after COBie 2.4 Job
seed/service_events.jsonl     the latest service per asset, after COBie v3 Event
seed/visible_assets.jsonl, seed/visible_asset_types.jsonl
documents/originals/    the pages and manuals as fetched. Private, like the pull
documents/fetched.json  where each original came from, its sha256 and its link check
documents/documents.jsonl     one line per document: anchors, clearance class, date, link
documents/chunks.jsonl        the English passages the index will hold
documents/translations.jsonl  each translated passage beside its original, and what made it
documents/summaries.jsonl     one summary per document, what it was made from and what made it
documents/vectors.jsonl       each chunk's and each summary's embedding
manifest.json           input hashes, what produced each part, counts, the sourced-clearance
                        share, the inventory's seed, fill rates and attribute catalogue,
                        and for the documents: links, translations, summaries and
                        ingestion spend. And the default origin, if the venue
                        definition names one
```

Run it again and it reuses the stored pull, originals, translations, summaries and
vectors, calls nobody and writes the same bytes. `--refetch` replaces the pull, `--refetch-documents`
fetches the originals again. If the stored pull or an original no longer matches the
hash it was pinned with, the build stops.

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

## Documents

The snapshot's prose comes from a corpus: a published JSON file listing sources, each
with a URL, a language and what it is about. Text is never in the corpus. Build fetches
it. KIT's is `packages/ingest/src/spatial_ingest/corpora/kit-campus-sued.json`: the 36
lecture hall pages of KIT's Studium und Behinderung office, each keyed by the building
number the page itself starts with. `--documents <file>` builds from another corpus. A
venue with no corpus gets no documents and needs no model.

It lists no manuals yet. I have not found manual links for the generated types that I
would trust, so that half is built and tested but the real corpus is pages only. Add a
manual as a `"kind": "manual"` entry anchored to its `asset_type` ids.

A document has one or more anchors, each a space, an asset type or one asset. Build
resolves them against the snapshot and drops what it cannot find, so a manual is only
ingested for a type that is actually placed, and its file is not even fetched otherwise.
A source with no anchor left is skipped, and the manifest says which and why.

Originals stay in `documents/originals/` and nothing ever serves them. A chunk carries
the original's URL, its language and the sha256 of the passage's original text, and
that is all that leaves the bundle. The German is in `translations.jsonl`, also private.

Build needs these for a corpus with anything in it:

```
AZURE_OPENAI_ENDPOINT            the Azure OpenAI resource
AZURE_OPENAI_API_KEY             its key
AZURE_OPENAI_API_VERSION         optional, 2024-10-21 by default
SPATIAL_COMPOSITION_DEPLOYMENT   translates and summarises. The same deployment that
                                 composes answers
SPATIAL_EMBEDDING_DEPLOYMENT     embeds the chunks. The RAG pipeline's, for now
```

| Rule | What it does |
| --- | --- |
| `anchor-building-by-ref@1` | A building number in the corpus is the building space whose sourced `ref` is that number. The anchor is flagged derived. No such building, no anchor. |
| `text-from-source@1` | The text of an HTML page, a PDF or a plain text file. HTML keeps what a reader sees, one line per block element. Where the corpus names two markers, only what lies between them is read, which keeps navigation and footers out. |
| `chunk-by-paragraph@1` | Whole paragraphs packed into passages of at most 1200 characters. A longer paragraph is cut at line ends, then sentence ends. |
| `translate-to-english@1` | A passage that is not English is translated by the composition deployment. This is the prompt's version, not a rule: the words are in `models.py`, and the manifest records the deployment, the model and this version for every translated document. |
| `numbers-in-words@1` | In the English text, a clock time ("16:30" is "sixteen thirty", and both ends of "8:00-16:00"), a floor ordinal ("2nd floor" is "second floor", "floor 2" is "floor two") and a small standalone count up to twenty ("2 steps" is "two steps") are written out. A measured value with its unit stays as digits ("90 cm", "37.1 MHz", "5 minutes", "5 psi", "2 ft"), and so does a number that names something ("room 7", "channel 2", "10.11"). |
| `date-from-last-change@1` | A page's document date is the day its own "letzte Änderung:" line says. No line, no date: the document is undated. Manuals are never dated. The day it was fetched is never its date. |
| `document-clearance@1` | A document's clearance class is its strictest anchor's. A manual shared by several types takes the least restrictive of those types first. An anchor with no clearance of its own (an untagged building) limits nothing. It lives in `packages/core`, with the other clearance rules. |

Fetching an original is also its link check: the source answered or it did not, and
the manifest and every chunk record `online` or `offline` with the day build asked. A
source that goes offline later keeps the original the bundle already has, so it is
still quoted. One that never answered has nothing to quote and is skipped.

Spend is tokens per deployment times a dated rate from
`packages/ingest/src/spatial_ingest/rates.json`, the same rows as the site's budget
table. A deployment with no row is recorded with its tokens and no cost, and the build
says so. Right now that is the translation deployment: I have no price for it yet.
Summaries run on the same deployment, so they are tokens without a cost too.

### Summaries

Every document gets one summary of at most 60 words, written by the composition
deployment from the document's own English passages and its title. This is SAC,
summary-augmented chunking, from the retrieval research (decision ticket 43 on the
map). A summary does two things:

- It goes in front of each of its document's passages when the passage is embedded:
  title, summary, passage. The passage's text does not change. Only the vector knows.
- It goes in the index as a chunk of its own with `role: summary`, carrying its
  document's anchors and clearance class. Passages have `role: passage`.

A summary is never evidence. It is there to find passages with. It is not quoted, not
cited, not checked for entailment and not shown. Retrieval hands back passages only.

Summaries are built one clearance class at a time, lowest first, and a summary only
ever reads passages of one document, which all share a class. The summariser refuses
anything else, and load checks the bundle's record again: every summary's sources must
be passages of its own document and its own class, or the snapshot fails. So what a
visitor's search can match never came from a staff or maintenance document.

A document longer than about 20,000 characters is summarised window by window, then
the part summaries are summarised into one. Parts stay in the bundle.

`documents/summaries.jsonl` keeps each summary with its sources, the hash of what the
model read, the deployment, the model, the prompt version (`summarise-document@1`, the
words are in `summaries.py`) and the tokens. A rebuild reads them from there. Load
never makes one. A summary made under another prompt version is made again.

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

Load also writes the document chunks and summaries to a search index, one per snapshot,
named after the venue and the date (`spatial-kit-campus-sued-20260915`). It finds the search
service in the environment:

```
AZURE_SEARCH_ENDPOINT   the Azure AI Search service
AZURE_SEARCH_API_KEY    an admin key, since load creates and drops indexes
SPATIAL_SEARCH_DIR      instead of those two: a directory, one JSON file per index
```

The directory is a stand-in for tests and laptops. It stores chunks and vectors and
ranks nothing. The Azure index is hybrid: text and title searchable for BM25, a vector
field for the embedding, and the anchor ids by kind, the flattened ancestor ids, the
clearance class and the role (passage or summary) all filterable. Load makes no model
call. The summaries and the vectors are the bundle's.

`ingest load --gold gold.json` also asks a question set of the new snapshot and prints
how it went. That never fails a load. See "On a freshly loaded snapshot".

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
- every chunk belongs to a document of the bundle and carries that document's anchors,
  no more and no fewer
- every chunk is anchored to spaces, types and assets of the snapshot
- every chunk's clearance class is its strictest anchor's, by `document-clearance@1`,
  worked out from the document's anchors and not from the chunk's own id lists. A chunk
  that dropped its strictest anchor and filed itself lower used to agree with itself
- every chunk's ancestor ids are its space and asset anchors' paths
- every summary was built from passages of its own document and its own clearance class
- every document in the index has exactly one summary, and it is one the bundle recorded
- the index holds as many chunks as the manifest says in all, and as many passages and
  as many summaries per clearance class

The chunk checks run over what the index gives back, not over the bundle. The link
check is reported in the log and never blocks.

A snapshot that fails any of them is recorded as failed in `control.snapshots` and
`promote` refuses it. Its schemas and its index stay, for looking at. Loading the same
bundle again rebuilds the schemas and the index from scratch, so it yields the same stores.

Load never moves the current-snapshot pointer (`control.current_snapshot`, one row), and
it refuses to reload the current or the previous snapshot, since conversations may be
held on either. Promote is one update of that row: the snapshot it names becomes current
and the one it replaces becomes previous. The row names the search index too, in
`search_index` and `previous_search_index`, beside the snapshot whose row in
`control.snapshots` names the schema pair. The next load drops every snapshot older than
both, index included; snapshots loaded but not yet promoted stay until they are older
than both too.

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
The two crossable rules are in `packages/core/src/spatial_core/circulation.py`, since
the service runs them.

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
| `visible-by-edges@1` | A space is visible to a persona when an edge it clears enters the space, or, with no edge entering, by the space's own clearance. Its ancestors are visible with it. Only edges through a door count. A snap shows nobody anything. |
| `snap-doorless-room@1` | An `indoor=room` with no door at all is joined to the nearest circulation space on a level they share, if their outlines come within 2 m. Two edges with no door, derived, for routes only. It joins a room to circulation, never two rooms through a wall. Rooms only: a corridor or an area with no door stays as mapped. |
| `snap-open-space-point@1` | A route destination that is a point in open space (a hydrant on the site) is taken to mean the nearest place a route can reach within 5 m. Derived, and flagged in the route with how far the point was moved. Worked out by the service per route, not stored. |
| `crossable-by-tags@1` | What a route may pass through by sourced tags: corridors and areas, stairs and lifts, footways, and rooms tagged hall, lobby, entrance, foyer or corridor. Also what a doorless room may snap to. |
| `crossable-by-door-count@1` | An `indoor=room` with no `room`, `amenity` or `office` tag and three or more doors may be passed through too. Derived, and flagged in every route that uses one, since it may still be an office. Worked out by the service from the persona's own edges, not stored: a door they may not take is not counted, so the same room can be a way through for staff and not for a visitor. |
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
- KIT's pages anchor to buildings only. The hall inside the building is not matched to
  a room: only about a third of the hall names match an OSM room name.
- A document's title is the corpus's and stays in the original language.
- `numbers-in-words` works off lists of units and label words. A unit that is missing
  gets its number written out. "in" for inches is missing on purpose, since "2 in the
  lobby" is a count. It also leaves a count alone when the sentence before
  it ended in a full stop ("Mind the stairs. 2 of them are loose"), since that looks
  like an abbreviation.
- If a model call fails midway, the build stops and the translations and summaries it
  already paid for are not kept. The corpus is 36 short pages, so I have not bothered.
- The summary prompt is mine, adapted from the prototype's. The +8 points of recall were
  measured with a local 7B model on another building's handover, not with this prompt,
  this model or this corpus. Nothing here measures it again.
- The 60 words are asked for, not enforced. The long-document window is counted in
  characters because build has no tokenizer.
- Changing the summary prompt version re-embeds every passage, since the summary is in
  front of it.
- An index loaded before summaries existed has no `role` field. Load the snapshot again.
- The real Azure AI Search client is only tested against a stand-in on localhost.
  Nobody has pointed it at a real service yet.
- An edge carries where its door is and the level it is on: the door's own `level` tag,
  else the one level its spaces share, else nothing. A door with no level changes no
  level in a route, so a stair door mapped without one makes the climb free.
- The snap only helps a room with no door at all. A room with one mapped door and a
  second unmapped opening gets nothing, and two corridor polygons that meet without a
  door node stay apart, always. The spec says doorless rooms only. On a campus mapped
  with few doors that will leave holes in the graph. Not measured.
- The footways between buildings are in the pull and make no spaces, so there is no
  graph outdoors. No route crosses from one building to another yet.
- The default origin is a door node typed into the venue definition (`default_origin`).
  The build checks it opens from outside into one mapped space and puts it in the
  manifest. KIT's definition names none yet: picking one needs a pull to look at.

## The service

One operation, `POST /turn`, plus `GET /venue` for what a caller needs before it asks
anything (below) and `GET /eval` for the last evaluation run's figures (see "Evaluation
runs and the eval snapshot"). `POST /turn` takes a persona, a snapshot
(none means the current one), the conversation's prior state and a question, and returns
a turn:

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
AZURE_SEARCH_ENDPOINT            optional: the Azure AI Search service, with AZURE_SEARCH_API_KEY
SPATIAL_SEARCH_DIR               optional, instead of those two: the stand-in index directory
SPATIAL_SEMANTIC_RANKER          optional: "on" reranks passages with the semantic ranker. Off by default
SPATIAL_EVAL_SNAPSHOT            optional: the file an evaluation run wrote its eval snapshot to
```

With no search index named, the service still runs and no document is ever retrieved.

### The venue

`GET /venue` says which snapshot is served, so a caller can pin a conversation to it
before the first turn, and what an answer is modelled on:

```
{"snapshot": "2026-09-15", "previous_snapshot": "2026-09-01",
 "sourced_clearance_share": 0.12,
 "personas": ["visitor", "staff", "maintenance"],
 "route_constants": {"walking_speed_m_per_s": 1.34, "stairs_s_per_level": 12.0,
                     "lift_wait_s": 40.0, "lift_s_per_level": 4.0, "storey_height_m": 4.0}}
```

`previous_snapshot` is null when nothing was served before the current one.
`sourced_clearance_share` is the current snapshot's manifest `clearance.sourced_share`,
and null when the manifest has none. It is not worked out again here. The route constants
are the ones routes.py computes with, under longer names: change one there and it changes
here. It never calls a model, so it costs nothing.

When there is nothing to say it answers 503 with a refusal body, the same shape as
`POST /turn`'s: `{"detail": "...", "code": "no-current-snapshot"}` when no snapshot is
current, `{"detail": "...", "code": "store-unreachable"}` when the database cannot be
connected to. The code is what to act on. The detail is for reading and never holds the
database URL.

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
| `empty-retrieval` | Document retrieval returned nothing. It never says why: no document there, none within clearance, or nothing to anchor to. |
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

Five question shapes run on five tools. `place-location` is "where is room X".
`asset-containment` is "what is installed in room X" and "where are the lights on floor
2". `asset-aggregation` is "which floor has the most lights". `attribute-filter` is
"which lights have no serial number" and "what falls due for service before June".

| Tool | What it does |
| --- | --- |
| `resolve_place` | One place a mention names. As above. |
| `resolve_asset_type` | The visible asset types a mention names: an exact name, else every word of the mention in a type's name ("lights" is every type of light). A set, so nobody is asked which. |
| `assets_in_scope` | Every visible asset under one scope node, or under the whole venue, of the given types if any, passing the given attribute filters if any. One step and one pass over ancestor paths, however many spaces lie under the scope. No call per space. |
| `aggregate` | Those assets counted per space of one tier under the scope. |
| `resolve_asset` | The one asset a follow-up reference points back at. See Follow-ups. It does not find an asset by name. |

That fifth tool came with the fifth shape: `asset-location` is "where is that one", and
ends in `resolve_asset`.

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

A count also says what it counts, and the host says that too. Where a segment does not
name the type itself, the type's name goes in after the number, with its citation:
"Medical-Dental Clinic has 28 Lighting and Appliance Panelboard". A count over several
types, or any, gets "assets". The model used to have to supply the noun, and on the
test bench it mostly didn't: the first real run came back with "has 1 1" and "has 2
Medical-Dental Clinic". So a count placeholder run straight into another placeholder
is rejected now, unless that one is the type it counts. No plurals for type names,
since "Equip-A1155" has none, so it reads a bit stiff. And the noun is English whatever
language the answer is in. Rough, but nothing a count says is the model's guess anymore.

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
as before. The gap turn still leaves its filter in the state.

A filter also works under `aggregate` ("which building has the most lights with no
service event"), in an `asset-aggregation` plan.
### Routes and proximity

The two lowest-priority shapes. `route` is "how do I get to the Audimax", `proximity`
is "where is the nearest extinguisher to room 107". Three more tools
(`packages/service/src/spatial_service/routes.py`):

| Tool | What it does |
| --- | --- |
| `places_matching` | Every visible place a description fits, as a set: by what the room is for (`room`, `amenity`), else by the words of its name. "The nearest toilet" is all the toilets. Only a route reads it. The word list is short and typed in. |
| `route` | The quickest route to a destination: one place, a set of places, or assets. It ends at the nearest. From a place, from where an earlier route ended, or from the default origin. |
| `nearest` | The nearest of some assets to a place, or to the default origin. |

**The graph is the persona's.** It is built from the edges the persona clears, between
spaces they can see. Nothing else is in it, so a route never touches a space beyond
clearance and cannot say one is there. The trace records how big the graph was.

**A route minimises modelled duration.** Nothing is measured:

```
walking   1.34 m/s everywhere                Weidmann 1993, a literature mean
stairs    12 s per level, half for a half    a 4.0 m storey at 0.326 m/s vertical
lift      40 s to take one, plus 4 s a level design figures, not measurements
storey    4.0 m                              a chosen constant
```

Every route comes back with whole seconds, whole metres and those constants, in
`routes` on the turn. Each one stands on its own: both ends, every space passed
through with its seconds and metres, and everything derived it leaned on. Rounding
further and wording are the page's. In the answer text a duration fills in as
"41 seconds (modelled)", by the host, so the word cannot be left out.

Metres are what is walked on the flat. A level change adds time and no distance.
Inside a space the walk is door to door in a straight line, so an L-shaped corridor
comes out short.

**Lifts are in by default.** So a floor only a lift reaches is routable. Stairs win
anyway for a level or two. `step_free` on the route step takes every stair out. The
planner sets it only when the question asks, and nothing in the host ever sets it
from the persona. The spec calls it a constraint. Here it is a flag on the step: the
constraint list is all "this mention within that one" and a flag with no mentions
did not fit it.

**What a route may cross.** Circulation by its sourced tags, plus an untagged room
with three or more doors, which is flagged `derived` in the route wherever one is
used. Only doors the persona may take count. A door into a room they cannot see is
not theirs, so it cannot turn a room into a way through for them. A room with a
function that is not circulation is never crossed, three doors or not. The rooms a route starts and ends in are always enterable. A doorless room is
reached by its snap, flagged too.

**A destination is places before anything is routed.**

- A named place is that place.
- A description is a set, and the route reaches the nearest. Ask for alternatives and
  the next nearest come back too.
- An asset is its space. An asset hung on something bigger than a room (a hydrant on
  the site) goes by its position: the mapped rooms the point lies in, and if it lies
  in none, the nearest place a route can reach within 5 m.
- Typed coordinates are never read. The request has no field for a point either, so
  asset positions are the only points there are so far.

**The open-space snap is 5 m.** Chosen, not measured, since there is no bundle here
to measure on. About a corridor and a wall: enough for a hydrant on a facade or a
point dropped just outside a door. It hides at most about four modelled seconds. The
snap distance is reported with the route and not added to its metres.

**The origin defaults.** A route step with no origin starts at the venue's default
origin, one entrance shared by every persona.

**No route is one abstention.** `no-route`, with the same words, whether a restricted
corridor is in the way, the crossing rule stopped it, the mapping has a gap, the
venue names no default origin, the destination came to no place or the only way is
by stairs and step-free was asked for. The trace gives no reason either. It says the
step found nothing and stops. A place the persona cannot see never gets that far. It
is `no-such-place`, as everywhere else.

**A plan holds at most two route steps, each with at most three alternatives.** Over
either and the plan is rejected.

**Proximity** is by modelled duration when a route reaches any of the assets, and the
answer carries that route. When none does, it is by shared containment: the assets in
the smallest space that also holds the anchor (same room, then floor, then building).
The trace says which, and the answer does not call the second one a distance.

**Follow-ups.** A route answer leaves the last route in the state (below). "And from
there to the library?" is a route whose origin is a follow-up reference to the place
the last one ended at. A route or a nearest step can also read the one asset a
follow-up points at ("the nearest extinguisher to that one"), which stands for its
space. The planner is told whether the last route was step-free and asked to keep it
when the question carries on. That is the one way step-free outlives its question, and
it still never comes from the persona.

Rough, all of it. Not measured against anything, and the spec lists both shapes as
unmeasured. A space with no outline cannot be routed to. Type and description matching
are word lists.

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
| `clarification` | A step matched several places, or the mention was a bare room number, which is never a label. Or a follow-up reference pointed at more than one asset. The candidates are all visible to the persona, at most six. The state carries them, and the next turn's planner sees their labels. |
| `abstention` | `no-such-place` when nothing the persona can see matches. A place beyond clearance and a place that does not exist give the same abstention, and the same trace but for the words asked. `no-such-asset-type` is the same for a type. Also `nothing-at-scale-level`, `step-dependent-question`, `unsupported-question`, `plan-rejected` and `composition-rejected`. `nothing-referred-to` when a follow-up reference has nothing to point at. `no-document` and `ungrounded` for document questions (see Documents in an answer). `nothing-recorded` is a room or floor with nothing in it, inside a building that has installations: no reason given. And `coverage-gap`, with the turn's `gap` saying which: `no-recorded-installations`, when nothing the persona can see is installed in a building or anything bigger, or `attribute-not-recorded`, when no asset asked about carries the attribute filtered on. A gap in the data, not a count of zero. Nothing else gives a reason. `no-route` is any route that cannot be found, whatever stopped it, and `nothing-near` a proximity question with nothing to be near. |
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
snapshot is neither current nor previous. That refusal is a 409 with
`"code": "snapshot-dropped"`, and a snapshot id that never existed gets the very same
one. The caller ends the conversation on that code, it does not parse the message.

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
- The filters in force are told to the planner, which has to repeat them in its plan
  for a follow-up to stay filtered. Code does not apply them by itself, and no real
  model has been asked to yet.
- There is no test for a value ("made by X"), only for presence and for dates.
- Clarifications, abstentions and coverage gaps are worded by the host, not composed.
- The plan schema is a plain union of step types, because strict structured output
  refuses `oneOf`. Whether the real deployment takes it has not been tried.

### Documents in an answer

Two more shapes, one more tool. `asset-document` is "what does the manual say about
servicing the unit in this room". `place-document` is "what does the Audimax's page say
about access". Both end in `retrieve_documents`, which takes anchors from an earlier
step and a query the planner wrote:

```
resolve_place -> resolve_asset_type -> assets_in_scope -> retrieve_documents   asset-document
resolve_asset -> retrieve_documents                                            asset-document
resolve_place -> retrieve_documents                                            place-document
```

The second one is the follow-up: "what does its manual say", about the one asset an
earlier turn found. `resolve_asset` reads conversation state, so an asset the persona
cannot see is not there to point at and the turn ends in `nothing-referred-to` before
anything is searched. The answer opens with the asset's room and a count of one, like
any asset-document answer. It reads stiff.

The anchors are ids the persona can see, and nothing else:

| Anchored by | The ids the filter names |
| --- | --- |
| assets | the assets the step found, and the types of those assets |
| one asset | that asset and its type |
| a place | the place and every visible space under it, the visible assets under it, and their types |

The filter the index gets is: clearance class within the persona's, and at least one of
the chunk's anchors among those ids. The spec words it as three clauses (clearance, scope
matches or a type is visible in scope, an anchor in the visible set). Spelling the scope
out as the visible ids in it folds the last two into one. It is a bit stricter in one
corner, on purpose: a document tied to the scope only through something the persona
cannot see does not come back, even if another of its anchors is visible elsewhere.

A type only gets into those ids through a visible asset of it. So a manual anchored to
a type is retrievable through such an asset and no other way, and a manual for
panelboards does not exist for a visitor.

The search is hybrid: keywords and the query's embedding, fused with RRF by the index.
The query is embedded by the deployment the snapshot's manifest names, the one that
embedded the chunks. The filter runs in the index before anything is ranked, so a
document the persona may not read cannot take a slot and shorten the answer. The host
checks every hit against the same filter anyway. The semantic ranker is a switch
(`SPATIAL_SEMANTIC_RANKER`), off by default. The stand-in index has no ranker, so there
the switch changes nothing but the trace.

Summaries are searched too and never returned. A summary that matches counts for its
document: each of that document's passages gets the summary's rank added to its own.
If none of them was among the candidates, they are fetched. Then every summary is
dropped, before anything leaves the tool. No summary text reaches a model, an answer,
a citation or the trace. The trace says `via_summary` on a passage whose summary
outranked it, and that is all.

Facts and passages go to composition as two lists and are never scored against each
other. The composer gets a passage's kind, title and text under a ref (`p1`, `p2`). It
does not get its id, link or date. A passage ref goes in a segment's `cites`, never in
braces, so nothing of a passage is filled in and a number still only comes from a fact.
The digit rule holds for document segments as well, quoted or not.

What the host checks of a document segment, and drops the segment for, alone, with a
note in the trace's `dropped_segments`:

- it cites a passage that was not retrieved this turn
- it quotes a span (in double quotes) that is not in a passage it cites, word for word.
  Whitespace is the only thing folded.
- it names a fact and cites a passage too. A segment rests on facts or on passages,
  never both. "MAIN MECHANICAL ROOM is within Second Floor" citing a floor plan is a
  fact the plan can't back, and the NLI check rightly drops it. That was most of the
  first run's failures. It's dropped here now, before anything scores it.
- it says what the documents don't say ("the documents do not say anything about X").
  Nothing entails an absence. That one is a short list of wordings, not a parser, so a
  new way of saying it will get through until someone adds it.

If no segment citing a passage is left, the turn is an abstention, `ungrounded`. One
gap: a fact the answer has to name counts as named even when the segment naming it
was dropped, same as before. So an answer can lose its total that way. Hasn't bitten
yet, and the prompt now tells the model to keep the facts in segments of their own.
Whether the passage backs what a segment says is not checked here. That is entailment,
the backend's half. Each segment therefore carries `passages`: every passage it cites
with its full ingested English text, the spans it quoted, and how to cite it.

- A translated passage is quoted in translation. `translated` and `language` say so and
  `url` is the original's.
- A source that was offline at build has `url: null`, with `link_status` and
  `link_checked_on` giving the day build looked.
- The document's date is in the trace (`trace.documents.retrieved`), next to what was
  searched: the query, the index, how many ids the filter named, whether the semantic
  ranker was on.

Retrieval that returns nothing is a trigger, `empty-retrieval`, and buys the one
recovery call. A second time, the turn abstains with `no-document`. A document beyond
clearance and a document that does not exist end the same way.

A turn also reports `embedding_calls`: the deployment and tokens of each query
embedding, apart from `llm_calls`. It is paid for, but it is not one of the three model
calls.

Rough edges:

- The Azure query client has never met Azure. Its request is from the REST reference
  and the tests only pin what it sends. The semantic configuration is new on the index,
  so an index created before this needs a reload before the ranker can be switched on.
- For a place high in the tree the filter names every visible space and asset under
  it. That is a long request for a whole campus. Not measured.
- A quote cannot hold a digit, because nothing a model writes can. "Ramp 90 cm wide"
  can be paraphrased around and not quoted.
- The quote check is exact. A model that tidies a comma loses the segment.
- Only double quotes count as quoting. Single quotes are not checked.
- An empty retrieval says nothing of why, so a second plan can only guess. It mostly
  costs a planning call and ends in `no-document`.
- How a summary's rank is folded in is my own guess at RRF over two rankings. Not measured.
- The real corpus has no manuals yet, so `asset-document` only finds anything on the
  mini campus fixture.
- Whether the composing model quotes cleanly, or the planner writes a useful query, is
  untested. Every test stubs both.

### Follow-ups

The service keeps nothing between turns. The state comes in with the request and the
new one goes back out. It holds:

| Field | What |
| --- | --- |
| `questions` | The text of earlier questions, oldest first, the last three. Composed answers are not carried. |
| `place` | The last place a turn resolved: the one located, the scope looked under, or the one space that came out on top of a count. |
| `asset_types` | The asset types the last asset question asked for. |
| `assets` | The assets the last containment answer rested on, if there were six or fewer. An answer over more leaves none: nobody can point at one of forty. |
| `since` | For each of those three, the turn that set it. |
| `constraints` | The constraints in force: one per attribute filter of the last asset question, kind `attribute-filter`. Scope and types as ids, the filter (attribute, test, date) in `args`. An asset question replaces them, with none if it had no filter. Any other turn leaves them. The next planner hears them as `conversation.filters`, minus any whose attribute is not in the persona's catalogue. |
| `route` | The last route, kind `route`: its ends as space ids, origin first, and `step_free` in `args`. A default origin is not an id, so then only the destination is there and `args.default_origin` is true. Seconds and metres are not carried, a follow-up routes again. The next planner hears it as `conversation.last_route`, by name. A route answer replaces it, any other turn leaves it. |
| `pending` | The candidates a clarification offered, their kind (place or asset) and the turn that offered them. Its `mention`, the words that were ambiguous, is capped at 200 characters like a plan's own mention. Longer and the request is refused. |

A turn that resolves nothing for a slot leaves it alone. So a third question can reach
back past the second, and the trace says which turn it reached.

A follow-up reference is a mention the planner marks with `refers`. The model says
where it points and what kind of thing is meant. Code finds what:

| The words | In the plan | Resolves to |
| --- | --- | --- |
| "that one" | kind `asset`, `{"to": "earlier-turn"}` | The asset in state. Several is a clarification. |
| "like it" | kind `asset-type`, `{"to": "earlier-turn"}` | The type of the asset in state. Assets of several types is a clarification over the assets. No asset held, then the types in state. |
| "there" | kind `place`, `{"to": "earlier-turn"}` | The place in state. |
| "the second one" | any kind, `{"to": "candidate", "ordinal": 2}` | The second candidate the pending clarification offered. Read as an asset type, that asset's type. |

A step over such a mention reads state and looks nothing up. The trace lists each one
under `references`: the words, the kind, where it pointed, the outcome, the ids and
names it came to, the turn those are from, and for "like it" the asset it went through.

Nothing to point at is a `nothing-referred-to` abstention, and buys no recovery call.
An ordinal past the last candidate offers the same candidates again. An asset candidate
carries its id, the room it is in and whether its position is recorded, since two
lights of one type in one room share everything else.

None of the incoming state is trusted. Before anything reads it, every id is checked
against what the persona can see in this snapshot, and whatever fails is dropped. A
tampered id, a stale one and one a persona was handed but cannot see all go the same
way, and the turn that follows is identical to one where the id was never there. A
constraint or a route is dropped whole if any id in it fails. Dropped candidates go
before ordinals are counted.

Rough here:

- An asset can only be reached by pointing back at it. "Where is Light Fixture B1 10"
  is a rejected plan.
- The planner has to work out the ordinal for "the one in 30.10" from the candidate
  labels. The old way, copying the label into a mention, still runs for places.
- "Those", meaning several assets at once, has no tool to feed. It asks which one.
- Whether the planning model writes `refers` reliably is untested. Every test stubs it.

## The test bench

Quality figures are not measured on the campus. They are measured on a real building
handover: a COBie workbook, the IFC models it was exported from and the O&M documents
handed over with it. That is the test bench. It goes through the same build, the same
bundle, the same load and the same invariants as the demo venue, into a database and an
index of its own.

```
uv run ingest fetch-handover --to handover/        # the pinned files, about 250 MB, once
# OCR the documents yourself into handover/ocr/<file name without .pdf>.txt
uv run ingest build --venue test-bench --date 2026-09-15 --handover handover/ --out bench-bundles
uv run ingest load --date 2026-09-15 --out bench-bundles --database-url <the bench's database>
```

The files are pinned by sha256 in `packages/ingest/src/spatial_ingest/venues/test-bench.json`,
the way the Overpass pull is pinned. Build stops if one does not match. No test and no
CI job fetches them. Tests run on a small hand-made handover in
`packages/ingest/tests/handover_fixture/` (13 rooms, 35 components, 12 documents), written
by `packages/ingest/tests/mini_handover.py`.

Source: BSI (2020) "Medical-Dental Test Files," buildingSMART International,
https://github.com/buildingsmart-community/Community-Sample-Test-Files (CC BY 4.0). Read
into the layer's records by the rules below. Nothing of it is in this repository or
redistributed.

A database holds one venue. Load refuses a bundle whose venue is not the one already in
the database, so the bench cannot land in the demo venue's stores or the other way round.
The index is named after the venue (`spatial-test-bench-20260915`).

| Rule | What it does |
| --- | --- |
| `tree-from-cobie-sheets@1` | The Facility is the building and the root. Every Floor row is a floor under it, whatever its category. Every Space is a room on its FloorName. |
| `room-label@1` | The same rule as the campus. A Space's Description is its name, its Name is its number: "CORRIDOR", or "PN 0001 room 1A01" where there is no description. |
| `clearance-from-space-category@1` | A room's clearance from its OmniClass Table 13 category, by code. Corridors, stairs, restrooms, waiting, reception and vestibules are visitor. Electrical, mechanical and equipment rooms, service spaces, hazardous storage, the loading dock and shafts are maintenance. Anything else, a missing category too, is staff. It is not the campus's rule, on purpose, and every one is an authored default: a handover has no access data. |
| `clearance-from-omniclass@1` | Asset types use the campus's Table 23 rule, unchanged. |
| `asset-in-first-space@1` | A Component listed in several spaces (a door in both rooms it joins) sits in the first. |
| `position-from-cobie-coordinate@1` | A Component's point from the Coordinate sheet, or the middle of its box. |
| `position-from-ifc-placement@1` | Else the placement of the IFC element whose GlobalId is the Component's ExtIdentifier. |
| `document-per-file@1` | One document per File, anchored to every row the Document sheet lists it for: a type, a component, a space, a floor, the facility. |
| `anchors-from-system@1` | A document keyed to a System is anchored to each of that system's components. Those anchors are flagged derived. |
| `document-kind-from-category@1` | Product data, O&M and maker's instructions are `manual`. Anything else is `handover-record`, a document kind only the bench has. |

Ids are the sheet's names, slugged: `space:1a01`, `asset:ac-unit-type-1-ac-1`,
`type:door-type-a`, `doc:panelboard-om-pdf`. Two names that slug alike stop the build.
"n/a" is COBie for nothing there and stays a gap.

Rough, or just not done:

- Only the workbook has been through this for real, on my machine: 269 rooms, 366 types,
  5,071 components, visible assets 1,136 / 4,502 / 5,071 per persona, which is what
  decision ticket 70 measured. The IFC reader and the whole document half have only ever
  run on the hand-made fixture. I have not downloaded the models or the 147 MB of
  documents, and nothing here has OCR'd anything.
- The bench's sourced half lands in the schema still called `snap_<date>_osm`, comment
  and all. The split is right (sourced there, seed empty), the name is wrong for it.
- A placement is a point in the model's own metres, put on the globe at an origin the
  definition makes up (0, 0), because the schema wants WGS84 and a redacted building has
  no address. Nothing prints or routes on it.
- The IFC reader is mine and small: placements only, read line by line. Around a minute
  for the four models, at a guess.
- A document with no OCR text reads as "offline" in the manifest. It is skipped either way.
- The Attribute sheet (door widths, finishes) is not read. Assets carry the six
  Component columns and nothing else.
- Job rows keep the sheet's own "n/a" where a frequency is blank, since the record
  wants a value there.
- Spaces have no outlines, the bench has no edges, and so no routes. By design.

### Gold, the question set and the leak checks

```
gold/            the gold scripts. Standard library only, not a package of the workspace
packages/bench   the harness: question set models, a scripted model, the checks, a CLI
```

Gold is what a correct turn would hold, worked out from the COBie workbook by scripts
that share no code with the layer. They are run as `python -I -S gold ...`, with
site-packages off, where `spatial_core` and friends cannot be imported at all. A test
holds them to that. The clearance tables are typed in again there from the decision
tickets. On the real workbook the two implementations agree on who sees what.

```
uv run bench gold cobie --workbook handover/2013-10-09-Clinic-Handover-v14.xlsx \
    --ocr handover/ocr --out gold.json
uv run bench check --bundle bench-bundles/2026-09-15 --gold gold.json \
    --admin-url postgresql://postgres:...@localhost/postgres --report checks.json
```

The question set is template questions per question shape, each asked as a visitor and
as a maintenance technician, each with its expected turn kind and its own gold: the ids,
the counts, the clarification candidates, the fill rate, the documents retrieval may
return. Nothing is drawn at random, so the same workbook gives the same file. Routes and
proximity have no questions and are listed as unmeasured. Follow-ups have none either.

`bench check` calls no model. The planner writes the plan a question's slots spell out
and the composer names what it is told to, so the tools, the clearance filter and the
host's checks are all real and the run is free. It loads the bundle into throwaway
databases (it needs a Postgres login that may create them, never the layer's own) and
runs three things:

- **Tool correctness.** Every turn against gold, by exact match on turn kind, ids and
  counts. Retrieved documents must all be ones gold says the persona may get.
- **Input leak check.** Nothing sent to the planner, the composer, the embedding
  deployment or the index filter holds an id outside the persona's visible set. It also
  looks for names only hidden things carry, and attributes only hidden assets carry. No
  model input here carries an id today, so ids alone would pass a service that showed a
  visitor the whole building by label. What the asker typed themselves does not count.
  It runs under all three personas. Gold is paired for the visitor and the technician,
  so staff are asked the same words with no gold behind them. A leak check needs none.
- **Counterfactual leak check.** The bundle is copied twice. Once with everything
  restricted removed: every space a visitor cannot see, every edge they may not take,
  every restricted asset and document. Once with more added: every restricted asset and
  document doubled, restricted types dropped into rooms a visitor walks through, a
  restricted twin of every document a visitor can read, and a hidden room behind three
  new doors next to rooms a visitor walks through. Each copy is loaded into its own
  stores and the visitor's questions are asked again. Every turn has to come back
  identical, whole, not just in kind, ids and numbers. On a venue with edges it also
  asks the visitor the way to up to 25 rooms they can see, on the bundle and on each
  copy. Routes have no gold, but this check only needs a turn to stay the same.

Each passes or fails. CI runs gold generation, the load invariants and all three on the
hand-made handover, and a failure fails the build.

The demo venue gets the same checks as an internal regression. It has no handover, so
its gold is read off its bundle (`bench gold bundle --bundle bundles/<date>`), visible
sets included. That checks the tools against the stores and not the clearance rules. CI
does it on the mini campus. Nothing from it is published.

Rough here:

- The scripted run on the real workbook passes all three, with 154 questions. That was
  without documents. With the fixture's documents it passes too. Nobody has run it on
  the real corpus.
- The counterfactual check is only as good as the stand-in index. That one scores
  keywords over the filtered passages. Azure scores BM25 over the whole index, restricted
  documents included, so a restricted document can move a visitor's ranking there
  without ever being returned. Not measured, and this check cannot see it.
- Gold only asks with words that name one thing: a label nothing else shares, a building
  and room number, a type's full name. Loose wording is for the hand-written questions.
- An attribute named like an OSM tag (`fire_extinguisher:type`) can be filtered on now:
  the filter's pattern takes the colon. Gold still skips those names, so only a service
  test covers it.
- The counterfactual check missed a real defect once: a door into a hidden room counted
  towards making a visible room crossable. It varied assets and documents only, and it
  asked no route. It does both now. The test bench has no edges though, so on the bench
  the route probes ask nothing. They only bite on a venue with doors, which here means
  the mini campus and a hand-drawn building in the tests.
- Staff turns are only held to the input leak check. No gold, so no tool correctness
  and no counterfactual for them.
- `within` (a place's enclosing spaces) is in gold and not compared. The trace does not
  carry a place's ancestors, only the composer's facts do, so there is nothing to compare
  with short of adding them to the trace.
- Timing leaks are not looked at.

### Evaluation runs and the eval snapshot

An evaluation run puts the question set to real models and writes down what came back.
It is the only thing in this repository that spends money on purpose, and it is started
by hand:

```
uv run bench run --gold gold.json --checks checks.json --rates rates.json \
    --ceiling 20 --out runs/
```

It reads the service's own environment (the database, both deployments, the index), so
point `SPATIAL_SERVICE_DATABASE_URL` at the test bench's database. It measures that
database's current snapshot. It also wants `SPATIAL_NLI_MODEL_DIR`, see below.

Before it calls anything it prints the run id, the snapshot, the deployments, the number
of questions, a cost estimate and the ceiling, and waits for you to type `yes`. There is
no flag to skip that. It refuses to start when:

- the estimate is over the ceiling
- a deployment it will call has no rate in the rates file
- the gold is not the test bench's (the demo venue's regression is never published)
- the checks report is for another snapshot, or skipped a leak check
- the baseline run asked another question set or another snapshot

`rates.json` is yours to keep current. This repository does not know what a deployment
costs. The numbers below are made up:

```
{"currency": "USD", "as_of": "2026-10-01",
 "per_million_tokens": {"gpt-5.4": {"input": 1.0, "output": 8.0},
                        "gpt-5.4-mini": {"input": 0.2, "output": 1.6},
                        "text-embedding-3-large": {"input": 0.1}}}
```

The estimate is a worst case: every turn plans twice and composes once, at assumed call
sizes (8000 in and 600 out for planning, 3000 and 400 for composition). Those sizes are
guesses, nothing has measured them. `--planning-tokens IN:OUT` and `--composition-tokens`
change them, and the run prints what it really used so the next estimate can be better.

The ceiling is what actually holds. Before every turn the run checks that what it has
spent, plus the dearest turn so far (or the estimate's turn, if that is more), still fits.
If not it stops, exits 3, writes its report and writes no eval snapshot: figures over the
questions it happened to reach are not figures over the question set. It also gives up
after five turns in a row with no model or no index behind them, and on the first turn
that calls a deployment with no rate. That turn is counted and scored, its unpriced
calls are listed in the report with no cost, and nothing more is asked. One turn much dearer
than every turn before it could still overshoot by the difference. Search request units
(the semantic ranker's) are not costed at all.

#### What is scored

Each turn is scored a stage at a time (`packages/bench/src/spatial_bench/score.py`). No
model judges anything.

- **Tools.** `bench check`'s comparison: turn kind, id sets, counts, fill rates, by exact
  match. A clarification's candidates have to equal gold's, and gold's are already cut to
  what the persona can see. That last one is also reported as a figure of its own.
- **Retrieval.** Recall@5 and MRR of the documents gold says the question is after, over
  the documents retrieved, in rank order. Only questions that are after a document count.
- **Composition.** Every number in the composed text is the number of a fact its segment
  cites (a count, a route's seconds, its metres), and every segment that cites a passage
  is entailed by it.

An answer is correct only when all three hold and the turn kind is the expected one.
Each question is scored against its own persona's gold, and the figures are given per
persona as well.

Results include a confusion matrix, expected turn kind against the kind produced.
Answered-when-it-should-have-declined (gold says abstention or coverage gap, the layer
answered) is counted on its own and should be zero.

Those two do not read the same thing, on purpose:

- The matrix is over what the site would send, after the NLI drop. An answer whose
  document segments the NLI model all drops is an abstention there.
- Answered-when-it-should-have-declined is over what the layer produced, before the NLI
  drop. The layer answered. The backend catching it afterwards does not undo that.

The run report has both kinds per question, `produced` and `layer_produced`.

Entailment uses the NLI model the site's backend runs,
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` as fp32 ONNX, with the same
rule: under P(entailment) 0.5 the segment fails. `SPATIAL_NLI_MODEL_DIR` is a directory
with the `model.onnx` and `tokenizer.json` the backend pins. The run records the model
file's sha256 in its deployments, so you can see it is the same file. It needs
`uv sync --extra nli` in `packages/bench`, which nothing else installs.

Decided here, since the spec does not say:

- Retrieval holds for a question when every document it is after is in the top five.
- A segment that cites several passages is scored against each and keeps its best score.
- The kind produced, in the matrix, is what the site would send. The backend drops
  segments the NLI model does not back and abstains when none is left, so an answer
  whose document segments all fail is an abstention there. See above for the one
  figure that reads the layer's own kind instead.
- The run id is the date plus eight random hex digits.

#### Hand-written questions

They live in `questions/hand-written.json`. There are 52, and a model wrote them, not a
person. I read them and liked them better than what I'd have written. They aim at words
no template uses ("loo", "elec room", "VAV boxes", "power outlets") and at clearance edges
(a visitor asking for the lift or the gas cylinders). Gold was worked out from the bench
bundle by script, not by the gold scripts, so a bundle bug would end up in both. The format:

```
{"format": "spatial-hand-written@1", "venue": "test-bench", "questions": [
  {"id": "boiler-room", "persona": "visitor", "shape": "place-location",
   "question": "Where would I find the boiler room?",
   "expected": {"kind": "abstention"}}]}
```

`expected` is the same as in the gold file (`spatial_bench.questions.Expected`): the kind,
and for an answer the ids, counts and documents you expect, written out by hand. Give
two questions the same `pair` when they are one question under two personas.

A run always reports them apart from the template questions, in the output and in the
eval snapshot. With none, their figures have `n` 0 and null values. That is a normal
snapshot.

#### What a run writes

Three files in `--out`:

- `run-<id>.json` (`spatial-eval-run@1`): the owner's copy. Every question with its
  stages, the estimate, the spend, the adoption rules. Gold ids are in the mismatches,
  so do not publish it.
- `segments-<id>.jsonl`: every segment the NLI model scored, one a line, with the full
  text of each passage it cites, the score against each, the best score and whether it
  was kept. It is there so a person can label segments and check the NLI model by hand.
  Private: it holds passage text of every clearance class, it is written readable by
  the owner only, and none of it goes into the eval snapshot or the metrics.
- `eval-snapshot.json` (`spatial-eval@1`): what the site reads. Only written when the run
  got through every question.

Spend is reported under the ledger category `evaluation`, by deployment and purpose, with
the rates' date. It is never visitor spend and the site's gate never reads it. The ledger
is the backend's. This is the run's report of what to book there. Nothing books it for
you. The eval snapshot has no money in it.

The confusion matrix is also pushed as a metric, `spatial.eval.turns`, to
`OTEL_EXPORTER_OTLP_ENDPOINT` when that is set. Labels: `run_id`, `snapshot`,
`planning_deployment`, `composition_deployment`, `embedding_deployment` ("none" for a
keyword-only run), `nli_deployment`, `question_set`, `expected_turn_kind`, `turn_kind`.
It is a series of its own and never adds to the live `spatial.turns`.

#### The eval snapshot

The service serves it at `GET /eval`, from the file `SPATIAL_EVAL_SNAPSHOT` names. The
contract is `packages/service/src/spatial_service/evaluation.py`, and
`packages/service/src/spatial_service/eval_contract/` holds one body of each kind for the
backend's contract tests to copy: `snapshot.json` (its numbers are made up) and
`empty.json`.

With no run, meaning no file, the answer is exactly this and nothing else counts as empty:

```
{"format": "spatial-eval@1", "run": null}
```

A file that is there and does not fit the contract is a 500 with
`"code": "eval-snapshot-unreadable"`. It is never turned into the empty answer.

A snapshot:

```
format        "spatial-eval@1"
label         "Measured on a test bench: a real building handover, not this campus"
run           id, date, venue, snapshot, semantic_ranker, gold_sha256,
              deployments {planning, composition, embedding (or null), nli}
figures       template and hand_written, each a section
leak_checks   input and counterfactual, each {passed, date}
unmeasured    ["routes", "proximity", "timing-leaks", "follow-ups"]
```

A section:

```
n                          questions in the set
correct                    a figure
stages.tools               a figure
stages.retrieval           recall_at_5 and mrr, a figure each
stages.composition         holds, numbers and entailment (over segments), a figure each
turn_kinds.matrix          {expected kind: {produced kind: count}}, {} when n is 0.
                           Produced kinds are the four plus "unavailable", as the
                           site would send them, after the NLI drop
turn_kinds.answered_when_it_should_have_declined   {count, of}, on what the layer
                           produced, before the NLI drop
clarification_candidates   a figure
by_persona                 {persona: figure of correct}
```

A figure is `{"value": 0.7532, "n": 154, "ci95": [0.6795, 0.8146]}`. Over nothing it is
`{"value": null, "n": 0, "ci95": null}`. Shares get a Wilson interval. Recall@5, MRR and
the paired differences get a percentile bootstrap with a fixed seed. The run block's
date, snapshot and deployments go for every figure in the file.

The leak checks' pass or fail and date come from `bench check`'s report (`--checks`).
The run does not rerun them. A failed leak check does not stop a run. It is published as
a fail.

Follow-ups are on the unmeasured list because the question set is one turn each. That
is one more than the spec lists.

#### Adoption rules

Worked out and printed at the end of a run, and written to the run report. Nothing
applies them.

- **Semantic ranker.** `--compare-ranker` retrieves each question a second time with the
  ranker on, next to the retrieval the turn uses. The ranker is to be adopted when the
  95% interval of the paired Recall@5 gain is wholly above zero.
- **Model pair.** `--baseline runs/run-<id>.json` compares this run with an earlier one
  on the same snapshot and question set, question by question. If this run was cheaper
  and the lower bound of its correctness difference is above minus five points, the
  cheaper pair may replace the baseline. Otherwise the baseline stays.

#### On a freshly loaded snapshot

`ingest load --gold gold.json` asks the question set of the snapshot it just loaded, with
the scripted model, and prints how many turns match gold. It is a report. Load exits 0
and the snapshot can be promoted whatever it says, and even when it cannot run at all.
`bench fresh --gold gold.json --snapshot <date>` does the same on its own, which is what
to use with the ingest image, since that image does not carry the bench. No figures come
out of either. Retrieval there is by keyword only.

Rough here:

- No real run has happened. Not one call to a real model, a real index or the real NLI
  model. Everything above is tested with a scripted model whose token usage the test
  sets, and an NLI stand-in.
- `OnnxNli` has never loaded the model. The label order comes from the model's
  `config.json` when the directory has one, and is assumed to be entailment first when
  not.
- The numbers check reads digits and holds each to a fact the same segment cites. The
  composer may not write digits, so every digit in an answer came from a placeholder. A
  number the model writes out as a word gets past it. It does not go back to the trace
  either: that a cited count is what the tools counted is the tools stage's to say.
- Tool correctness leans on the trace. A model that plans differently from the scripted
  one and still answers right could be marked wrong on a detail of the trace. Not seen,
  since no real model has been run.
- NLI precision is to be re-checked by hand on about 50 real composed segments. Nothing
  here does that. A run writes the segments to label, with passage text and scores, to
  `segments-<id>.jsonl`. The labelling and the comparing are still to do.
- Timing leaks are listed as unmeasured and that is all.

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

No test makes a paid call. The venue's pages, the manuals, Azure OpenAI and Azure AI
Search are all stand-ins on localhost (`packages/ingest/tests/stubs.py`), and the root
`conftest.py` points every test's search index at a temporary directory. No test fetches
the test bench's handover either: see The test bench.

## Data

Venue structure is © OpenStreetMap contributors, under the
[ODbL](https://opendatacommons.org/licenses/odbl/). The `osm/` half of a bundle is the
derived database, and the rules above are the method.

The accessibility pages are KIT's (Studium und Behinderung), which allows reuse for
educational, scientific or private purposes with the source named. They are translated,
quoted and linked. They are not OSM's and never go in the `osm` schema: chunks live in
the search index only.

The test bench is the buildingSMART Medical-Dental Clinic sample (CC BY 4.0), credited
under The test bench. It never shares a database or an index with the campus.
