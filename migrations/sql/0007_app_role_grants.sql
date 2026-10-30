-- The application connects as a least-privilege role (no superuser, no
-- BYPASSRLS, not the table owner), so row-level security genuinely applies.
-- Migrations run as the owner (DATABASE_MIGRATION_URL). Create the role
-- once:  CREATE ROLE agentspec_app LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS;
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agentspec_app') THEN
    EXECUTE 'GRANT USAGE, CREATE ON SCHEMA public TO agentspec_app';
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO agentspec_app';
    EXECUTE 'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO agentspec_app';
    EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO agentspec_app';
    EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO agentspec_app';
    EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM agentspec_app';
    EXECUTE 'GRANT EXECUTE ON FUNCTION ensure_monthly_partitions(text, int) TO agentspec_app';
  END IF;
END $$;
