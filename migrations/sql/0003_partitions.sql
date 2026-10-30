-- Append-only, high-volume tables are range-partitioned by month: retention
-- is "detach + drop the old partition" (instant, no vacuum storm) instead of
-- DELETE, and recent partitions stay small and hot in cache.

CREATE TABLE IF NOT EXISTS outbox_events (
  id varchar(48) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  tenant_id varchar(64) NOT NULL,
  topic varchar(64) NOT NULL,
  payload jsonb NOT NULL,
  published_at timestamptz,
  PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);
CREATE INDEX IF NOT EXISTS ix_outbox_unpublished ON outbox_events (created_at) WHERE published_at IS NULL;

CREATE TABLE IF NOT EXISTS audit_log (
  id varchar(48) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  tenant_id varchar(64) NOT NULL,
  actor varchar(128) NOT NULL,
  action varchar(64) NOT NULL,
  resource_type varchar(32) NOT NULL,
  resource_id varchar(128) NOT NULL,
  request_id varchar(64),
  details jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);
CREATE INDEX IF NOT EXISTS ix_audit_tenant_time ON audit_log (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_audit_action ON audit_log (action, created_at DESC);

-- Creates monthly partitions from last month to `months_ahead` ahead, plus
-- a DEFAULT partition so an insert can never fail for a missing month.
-- The outbox relay calls this daily.
CREATE OR REPLACE FUNCTION ensure_monthly_partitions(parent text, months_ahead int DEFAULT 3)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE m date; part text;
BEGIN
  FOR i IN -1..months_ahead LOOP
    m := (date_trunc('month', now()) + make_interval(months => i))::date;
    part := format('%s_%s', parent, to_char(m, 'YYYYMM'));
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   part, parent, m, (m + interval '1 month')::date);
  END LOOP;
  EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I DEFAULT', parent || '_default', parent);
END $$;

SELECT ensure_monthly_partitions('outbox_events');
SELECT ensure_monthly_partitions('audit_log');

ALTER TABLE outbox_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE outbox_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON outbox_events;
CREATE POLICY tenant_isolation ON outbox_events USING (tenant_id = app_tenant() OR app_is_system())
  WITH CHECK (tenant_id = app_tenant() OR app_is_system());
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON audit_log;
CREATE POLICY tenant_isolation ON audit_log USING (tenant_id = app_tenant() OR app_is_system())
  WITH CHECK (tenant_id = app_tenant() OR app_is_system());

-- Tamper-evidence: if the app connects as a dedicated role (recommended:
-- APP_DB_ROLE), it may only INSERT/SELECT the audit log.
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agentspec_app') THEN
    EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM agentspec_app';
    EXECUTE 'GRANT SELECT, INSERT ON audit_log TO agentspec_app';
  END IF;
END $$;
