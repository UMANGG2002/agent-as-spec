-- pgvector for mem0's memory store. Creating an untrusted extension needs a
-- privileged role; if the app role can't, a DBA runs this once.
DO $$ BEGIN
  CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN insufficient_privilege THEN
  RAISE NOTICE 'pgvector not installed: run CREATE EXTENSION vector as a superuser';
END $$;
