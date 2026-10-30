-- Row-level security: Postgres itself enforces tenant isolation.
-- Every app transaction runs `set_config('app.tenant_id', <tenant>, true)`
-- (app/db.py); cross-tenant maintenance jobs set app.role = 'system'.
-- FORCE makes the policies apply to the table owner too.

CREATE OR REPLACE FUNCTION app_tenant() RETURNS text LANGUAGE sql STABLE AS
$$ SELECT nullif(current_setting('app.tenant_id', true), '') $$;
CREATE OR REPLACE FUNCTION app_is_system() RETURNS boolean LANGUAGE sql STABLE AS
$$ SELECT coalesce(current_setting('app.role', true), '') = 'system' $$;

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['knowledge_documents','agent_specs','agent_spec_versions','tool_approvals',
                           'invocation_records','execution_plans','agent_definitions',
                           'notification_preferences','chat_sessions','invocations','tenants']
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I USING (tenant_id = app_tenant() OR app_is_system()) '
                   'WITH CHECK (tenant_id = app_tenant() OR app_is_system())', t);
  END LOOP;
END $$;

-- Grants are visible to both sides of the grant.
ALTER TABLE capability_grants ENABLE ROW LEVEL SECURITY;
ALTER TABLE capability_grants FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS grant_parties ON capability_grants;
CREATE POLICY grant_parties ON capability_grants
  USING (grantor_tenant_id = app_tenant() OR grantee_tenant_id = app_tenant() OR app_is_system())
  WITH CHECK (grantor_tenant_id = app_tenant() OR grantee_tenant_id = app_tenant() OR app_is_system());

-- Capabilities: readable if owned, published to the marketplace, or granted;
-- writable only by the owner.
ALTER TABLE capabilities ENABLE ROW LEVEL SECURITY;
ALTER TABLE capabilities FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS cap_read ON capabilities;
DROP POLICY IF EXISTS cap_write ON capabilities;
DROP POLICY IF EXISTS cap_update ON capabilities;
DROP POLICY IF EXISTS cap_delete ON capabilities;
CREATE POLICY cap_read ON capabilities FOR SELECT USING (
  tenant_id = app_tenant() OR visibility = 'marketplace' OR app_is_system()
  OR id IN (SELECT capability_id FROM capability_grants
            WHERE grantee_tenant_id = app_tenant() AND status = 'approved'));
CREATE POLICY cap_write ON capabilities FOR INSERT WITH CHECK (tenant_id = app_tenant() OR app_is_system());
CREATE POLICY cap_update ON capabilities FOR UPDATE USING (tenant_id = app_tenant() OR app_is_system())
  WITH CHECK (tenant_id = app_tenant() OR app_is_system());
CREATE POLICY cap_delete ON capabilities FOR DELETE USING (tenant_id = app_tenant() OR app_is_system());
