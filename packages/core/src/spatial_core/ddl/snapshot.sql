-- One snapshot's stores: a schema pair and the views that join them (ADR 0001).
--
--   {osm}    everything sourced from OSM or derived from it by a published rule.
--            The unit published under the ODbL.
--   {seed}   data generated so the demo has something to ask about. Not OSM's.
--   {views}  what the service reads. It never reads either schema directly.
--
-- Ingestion applies this to fresh schemas on every load. Placeholders in braces
-- are filled in by ingestion.

CREATE SCHEMA {osm};
CREATE SCHEMA {seed};
CREATE SCHEMA {views};

COMMENT ON SCHEMA {osm} IS
    'Derived from OpenStreetMap, (c) OpenStreetMap contributors, available under the ODbL.';
COMMENT ON SCHEMA {seed} IS 'Generated seed data. Not derived from OpenStreetMap.';

-- One row per space, field for field the Space record. Values that carry their
-- provenance ({{value, provenance, rule, basis}}) are jsonb.
CREATE TABLE {osm}.spaces (
    id                text PRIMARY KEY,
    record_provenance text NOT NULL CHECK (record_provenance IN ('sourced', 'seed')),
    source            text,
    tier              jsonb NOT NULL,
    -- No foreign key: a seed space may hang from an OSM one, across the pair.
    -- Load checks the tree over the views instead.
    parent_id         text,
    ancestor_path     jsonb NOT NULL,
    name              jsonb,
    ref               jsonb,
    label             jsonb,
    addressable       boolean NOT NULL,
    levels            jsonb,
    attributes        jsonb NOT NULL,
    outline           jsonb,
    -- {{persona, access, rule, authored_default}}: the raw tag beside what it mapped to.
    clearance         jsonb
);
CREATE INDEX ON {osm}.spaces (parent_id);

CREATE TABLE {osm}.spans (
    space_id   text NOT NULL,
    floor_id   text NOT NULL,
    provenance text NOT NULL CHECK (provenance IN ('sourced', 'derived', 'inferred')),
    rule       text NOT NULL,
    PRIMARY KEY (space_id, floor_id)
);

-- The circulation edges: one row per way through a door. An edge is sourced or
-- derived, never seed, but the seed half has the table so the views stay uniform.
-- A row with no door is a doorless room snapped to circulation, for routes only.
CREATE TABLE {osm}.edges (
    id              text PRIMARY KEY,
    door            text,
    from_id         text,
    to_id           text NOT NULL,
    door_clearance  jsonb,
    clearance       jsonb NOT NULL,
    provenance      text NOT NULL CHECK (provenance IN ('sourced', 'derived')),
    rule            text NOT NULL,
    -- Where the door is, and the level it is on.
    lon             double precision,
    lat             double precision,
    level           text
);
CREATE INDEX ON {osm}.edges (to_id);

-- What exists for each persona: one set per persona, computed at build.
CREATE TABLE {osm}.visible_spaces (
    persona  text NOT NULL CHECK (persona IN ('visitor', 'staff', 'maintenance')),
    space_id text NOT NULL,
    PRIMARY KEY (persona, space_id)
);

-- Asset types, after COBie Type. Values that carry their provenance are jsonb.
CREATE TABLE {osm}.asset_types (
    id                    text PRIMARY KEY,
    record_provenance     text NOT NULL CHECK (record_provenance IN ('sourced', 'seed')),
    name                  jsonb NOT NULL,
    omniclass             jsonb NOT NULL,
    manufacturer          jsonb,
    model                 jsonb,
    warranty_parts_years  jsonb,
    warranty_labor_years  jsonb,
    expected_life_years   jsonb,
    clearance             jsonb NOT NULL
);

-- Assets, after COBie Component. No foreign keys: a seed asset sits in an OSM
-- space, across the pair. Load checks that every asset sits in exactly one space.
CREATE TABLE {osm}.assets (
    id                text PRIMARY KEY,
    record_provenance text NOT NULL CHECK (record_provenance IN ('sourced', 'seed')),
    source            text,
    type_id           text NOT NULL,
    space_id          text NOT NULL,
    ancestor_path     jsonb NOT NULL,
    name              jsonb NOT NULL,
    door              text,
    position          jsonb,
    attributes        jsonb NOT NULL
);
CREATE INDEX ON {osm}.assets (space_id);
CREATE INDEX ON {osm}.assets (type_id);

-- Planned jobs per asset type, after COBie 2.4 Job.
CREATE TABLE {osm}.maintenance_tasks (
    id                text PRIMARY KEY,
    record_provenance text NOT NULL CHECK (record_provenance IN ('sourced', 'seed')),
    type_id           text NOT NULL,
    name              jsonb NOT NULL,
    category          jsonb NOT NULL,
    status            jsonb NOT NULL,
    description       jsonb NOT NULL,
    duration          jsonb NOT NULL,
    duration_unit     jsonb NOT NULL,
    start             jsonb NOT NULL,
    task_start_unit   jsonb NOT NULL,
    frequency         jsonb NOT NULL,
    frequency_unit    jsonb NOT NULL,
    task_number       jsonb NOT NULL,
    priors            jsonb,
    resource_names    jsonb
);

-- What was done to an asset, when and by whom, after COBie v3 Event. Nothing
-- about the outcome.
CREATE TABLE {osm}.service_events (
    id                text PRIMARY KEY,
    record_provenance text NOT NULL CHECK (record_provenance IN ('sourced', 'seed')),
    asset_id          text NOT NULL,
    task_id           text,
    name              jsonb NOT NULL,
    start             jsonb NOT NULL,
    "end"             jsonb,
    performer         jsonb NOT NULL
);
CREATE INDEX ON {osm}.service_events (asset_id);

-- The assets and asset types that exist for each persona, computed at build.
CREATE TABLE {osm}.visible_assets (
    persona  text NOT NULL CHECK (persona IN ('visitor', 'staff', 'maintenance')),
    asset_id text NOT NULL,
    PRIMARY KEY (persona, asset_id)
);

CREATE TABLE {osm}.visible_asset_types (
    persona text NOT NULL CHECK (persona IN ('visitor', 'staff', 'maintenance')),
    type_id text NOT NULL,
    PRIMARY KEY (persona, type_id)
);

CREATE TABLE {seed}.spaces (LIKE {osm}.spaces INCLUDING ALL);
CREATE TABLE {seed}.spans (LIKE {osm}.spans INCLUDING ALL);
CREATE TABLE {seed}.edges (LIKE {osm}.edges INCLUDING ALL);
CREATE TABLE {seed}.visible_spaces (LIKE {osm}.visible_spaces INCLUDING ALL);
CREATE TABLE {seed}.asset_types (LIKE {osm}.asset_types INCLUDING ALL);
CREATE TABLE {seed}.assets (LIKE {osm}.assets INCLUDING ALL);
CREATE TABLE {seed}.maintenance_tasks (LIKE {osm}.maintenance_tasks INCLUDING ALL);
CREATE TABLE {seed}.service_events (LIKE {osm}.service_events INCLUDING ALL);
CREATE TABLE {seed}.visible_assets (LIKE {osm}.visible_assets INCLUDING ALL);
CREATE TABLE {seed}.visible_asset_types (LIKE {osm}.visible_asset_types INCLUDING ALL);

-- Each table of the pair, joined.
CREATE VIEW {views}.spaces AS
    SELECT * FROM {osm}.spaces
    UNION ALL
    SELECT * FROM {seed}.spaces;

CREATE VIEW {views}.spans AS
    SELECT * FROM {osm}.spans
    UNION ALL
    SELECT * FROM {seed}.spans;

CREATE VIEW {views}.edges AS
    SELECT * FROM {osm}.edges
    UNION ALL
    SELECT * FROM {seed}.edges;

CREATE VIEW {views}.visible_spaces AS
    SELECT * FROM {osm}.visible_spaces
    UNION ALL
    SELECT * FROM {seed}.visible_spaces;

CREATE VIEW {views}.asset_types AS
    SELECT * FROM {osm}.asset_types
    UNION ALL
    SELECT * FROM {seed}.asset_types;

CREATE VIEW {views}.assets AS
    SELECT * FROM {osm}.assets
    UNION ALL
    SELECT * FROM {seed}.assets;

CREATE VIEW {views}.maintenance_tasks AS
    SELECT * FROM {osm}.maintenance_tasks
    UNION ALL
    SELECT * FROM {seed}.maintenance_tasks;

CREATE VIEW {views}.service_events AS
    SELECT * FROM {osm}.service_events
    UNION ALL
    SELECT * FROM {seed}.service_events;

CREATE VIEW {views}.visible_assets AS
    SELECT * FROM {osm}.visible_assets
    UNION ALL
    SELECT * FROM {seed}.visible_assets;

CREATE VIEW {views}.visible_asset_types AS
    SELECT * FROM {osm}.visible_asset_types
    UNION ALL
    SELECT * FROM {seed}.visible_asset_types;

-- The attribute catalogue, derived per snapshot: for each asset type, which
-- attributes at least one of its assets carries, and how many do.
CREATE VIEW {views}.attribute_catalogue AS
    SELECT asset.type_id, attribute.key AS attribute, count(*) AS assets
    FROM {views}.assets asset, jsonb_object_keys(asset.attributes) AS attribute (key)
    GROUP BY asset.type_id, attribute.key;

GRANT USAGE ON SCHEMA {views} TO {reader};
GRANT SELECT ON ALL TABLES IN SCHEMA {views} TO {reader};
