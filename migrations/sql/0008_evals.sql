-- Evaluation framework: sets, runs, per-case results, online samples.
-- Tables are created from the ORM metadata by the migrator (step 0001) on a
-- fresh or existing database; this file adds the tenant isolation.
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['eval_sets','eval_runs','eval_case_results','eval_samples']
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
    EXECUTE format('CREATE POLICY tenant_isolation ON %I USING (tenant_id = app_tenant() OR app_is_system()) '
                   'WITH CHECK (tenant_id = app_tenant() OR app_is_system())', t);
  END LOOP;
END $$;
