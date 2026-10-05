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
    outline           jsonb
);
CREATE INDEX ON {osm}.spaces (parent_id);

CREATE TABLE {osm}.spans (
    space_id   text NOT NULL,
    floor_id   text NOT NULL,
    provenance text NOT NULL CHECK (provenance IN ('sourced', 'derived', 'inferred')),
    rule       text NOT NULL,
    PRIMARY KEY (space_id, floor_id)
);

CREATE TABLE {seed}.spaces (LIKE {osm}.spaces INCLUDING ALL);
CREATE TABLE {seed}.spans (LIKE {osm}.spans INCLUDING ALL);

-- The seed half carries no spaces yet. Once it holds assets, they join the OSM
-- spaces by id here.
CREATE VIEW {views}.spaces AS
    SELECT * FROM {osm}.spaces
    UNION ALL
    SELECT * FROM {seed}.spaces;

CREATE VIEW {views}.spans AS
    SELECT * FROM {osm}.spans
    UNION ALL
    SELECT * FROM {seed}.spans;

GRANT USAGE ON SCHEMA {views} TO {reader};
GRANT SELECT ON ALL TABLES IN SCHEMA {views} TO {reader};
