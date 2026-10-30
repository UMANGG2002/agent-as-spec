-- Capability search: a generated tsvector over name/description/tool text
-- with a GIN index (full-text), plus trigram on the name (fuzzy/typo match).
CREATE EXTENSION IF NOT EXISTS pg_trgm;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS search_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', coalesce(search_text, ''))) STORED;
CREATE INDEX IF NOT EXISTS ix_capabilities_search ON capabilities USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS ix_capabilities_name_trgm ON capabilities USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_capabilities_market ON capabilities (visibility, id) WHERE visibility = 'marketplace';

-- Hot-path composite indexes (every one matches a WHERE in the code).
CREATE INDEX IF NOT EXISTS ix_grants_grantee_status ON capability_grants (grantee_tenant_id, status, capability_id);
CREATE INDEX IF NOT EXISTS ix_grants_grantor_time ON capability_grants (grantor_tenant_id, requested_at DESC);
CREATE INDEX IF NOT EXISTS ix_approvals_tenant_status ON tool_approvals (tenant_id, status, requested_at DESC);
CREATE INDEX IF NOT EXISTS ix_approvals_fingerprint ON tool_approvals (fingerprint, status);
CREATE INDEX IF NOT EXISTS ix_specs_tenant_updated ON agent_specs (tenant_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_kb_tenant ON knowledge_documents (tenant_id, created_at DESC);
