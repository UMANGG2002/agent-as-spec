-- Runs once when the Postgres volume is first created (docker-compose only).
-- Production: create the same least-privilege role through your DB tooling.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE ROLE agentspec_app LOGIN PASSWORD 'app_pw' NOSUPERUSER NOBYPASSRLS;
GRANT CONNECT ON DATABASE agentspec TO agentspec_app;
