psql -h /tmp -d postgres -c "CREATE ROLE nodetool WITH LOGIN PASSWORD 'nodetool_password';"
psql -h /tmp -d postgres -c "CREATE DATABASE nodetool OWNER nodetool;"
psql -h /tmp -d postgres -c "GRANT ALL ON SCHEMA public TO nodetool;"
psql -h /tmp -d postgres -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO nodetool;"
