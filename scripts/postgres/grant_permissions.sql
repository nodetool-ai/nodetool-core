-- PostgreSQL permissions setup for nodetool
-- Run this script as a PostgreSQL superuser (postgres)

-- If using development database with peer auth (no postgres superuser):
-- psql -U mg -d nodetool -f scripts/postgres/grant_permissions.sql

-- For production/development:
GRANT CREATE ON DATABASE nodetool TO nodetool;
GRANT USAGE ON SCHEMA public TO nodetool;
GRANT CREATE ON SCHEMA public TO nodetool;

-- For test database:
GRANT CREATE ON DATABASE nodetool_test TO nodetool_test;
GRANT USAGE ON SCHEMA public TO nodetool_test;
GRANT CREATE ON SCHEMA public TO nodetool_test;
