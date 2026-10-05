-- The control schema: which snapshots are loaded, and which one is current.
--
-- Ingestion applies this at the start of every load and promote. Every statement
-- is idempotent. Placeholders in braces are filled in by ingestion.

-- The service's role. It reads the control tables and each snapshot's views, and
-- nothing else. A login for the service is granted membership of it; ingestion's
-- own login owns every object and is the only one that writes.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = {reader_name}) THEN
        CREATE ROLE {reader} NOLOGIN;
    END IF;
END
$$;

DO $$
BEGIN
    EXECUTE format('REVOKE ALL ON DATABASE %I FROM PUBLIC', current_database());
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), {reader_name});
END
$$;

CREATE SCHEMA IF NOT EXISTS control;
GRANT USAGE ON SCHEMA control TO {reader};

-- One row per snapshot whose schemas exist. Only a snapshot whose latest load
-- passed its checks has status 'loaded', and only that can be promoted.
CREATE TABLE IF NOT EXISTS control.snapshots (
    snapshot     date PRIMARY KEY,
    venue        text NOT NULL,
    osm_schema   text NOT NULL,
    seed_schema  text NOT NULL,
    views_schema text NOT NULL,
    status       text NOT NULL CHECK (status IN ('loaded', 'failed')),
    failures     jsonb NOT NULL,
    manifest     jsonb,
    loaded_at    timestamptz NOT NULL
);

-- The current-snapshot pointer: exactly one row. Promotion is one update of it,
-- and the snapshot it replaces becomes the previous one, which load keeps.
CREATE TABLE IF NOT EXISTS control.current_snapshot (
    singleton   boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    snapshot    date REFERENCES control.snapshots,
    previous    date REFERENCES control.snapshots,
    promoted_at timestamptz
);
INSERT INTO control.current_snapshot (singleton) VALUES (true) ON CONFLICT DO NOTHING;

GRANT SELECT ON control.snapshots, control.current_snapshot TO {reader};
