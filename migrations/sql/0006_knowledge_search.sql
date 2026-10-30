ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS search_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED;
CREATE INDEX IF NOT EXISTS ix_kb_search ON knowledge_documents USING GIN (search_tsv);
