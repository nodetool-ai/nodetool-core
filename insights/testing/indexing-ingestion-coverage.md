# Indexing Ingestion Module Coverage

**Insight**: Added comprehensive test coverage for the document ingestion pipeline in `src/nodetool/indexing/ingestion.py`.

**Rationale**: The indexing module handles critical document processing operations (PDF/markdown conversion, chunking, workflow graph parsing) that had no test coverage. This is a core business logic component used throughout the system.

**Tests Added**:
- `tests/indexing/test_ingestion.py` (65 tests across 3 test classes)

**Coverage**:
- `Document` class: initialization with and without metadata
- `chunk_documents_recursive()`: single/multiple documents, custom chunk sizes, overlap handling, metadata preservation, empty/long text edge cases
- `chunk_documents_markdown()`: markdown header parsing, multiple documents, custom parameters, metadata preservation
- `find_input_nodes()`: graph parsing for CollectionInput, FileInput, DocumentFileInput nodes, empty/invalid graph handling

**Files**: 
- `tests/indexing/test_ingestion.py`
- `src/nodetool/indexing/ingestion.py`

**Impact**: 65 new tests covering critical document processing functionality with edge cases and error conditions.
