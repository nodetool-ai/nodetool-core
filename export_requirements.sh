#!/bin/bash

# Export poetry dependencies to requirements.txt
poetry export -f requirements.txt --only main --without-hashes -o requirements.txt 