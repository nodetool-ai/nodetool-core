# Docker Integration Tests for Self-Hosted Deployer

This directory contains integration tests that use real Docker to verify the self-hosted deployment functionality.

## Test Files

- **`test_self_hosted.py`** - Unit tests with mocked Docker commands (fast, no Docker required)
- **`test_self_hosted_docker_integration.py`** - Integration tests with real Docker (slower, requires Docker)

## Running Tests

### Run All Tests (Unit + Integration)
```bash
pytest tests/deploy/ -v
```

### Run Only Unit Tests (Fast, No Docker Required)
```bash
pytest tests/deploy/test_self_hosted.py -v
```

### Run Only Integration Tests (Requires Docker)
```bash
pytest tests/deploy/test_self_hosted_docker_integration.py -v
```

### Run Tests by Marker
```bash
# Run all integration tests
pytest -m integration -v

# Skip integration tests
pytest -m "not integration" -v
```

## Test Behavior

### When Docker Is Available
- All tests run normally
- Tests actually create/start/stop Docker containers
- Uses `nginx:alpine` as a lightweight test image
- Properly cleans up containers after each test

### When Docker Is Not Available
- Tests are automatically skipped with message: "Docker is not available"
- No failures, just skipped tests
- Exit code is still 0 (success)

Example output without Docker:
```
tests/deploy/test_self_hosted_docker_integration.py::TestLocalExecutor::test_execute_echo
SKIPPED (Docker is not available)
```

## GitHub Actions

The `.github/workflows/test.yml` workflow includes:
- Docker Buildx setup
- Docker image building
- All integration tests run with real Docker

## Test Coverage

### LocalExecutor Tests
- Real command execution
- Docker command integration
- Directory creation

### SelfHostedDeployer Tests
- Localhost detection
- Executor selection
- Directory management

### Docker Lifecycle Tests
- Container creation and removal
- Volume mounting
- Network configuration
- Health checks

### Cleanup Tests
- Container removal verification
- Resource cleanup

## Adding New Integration Tests

1. Create test class or function in `test_self_hosted_docker_integration.py`
2. Tests will automatically be skipped if Docker unavailable (due to module-level `pytestmark`)
3. Add cleanup in `try/finally` blocks to ensure containers are removed
4. Use the `docker_client` fixture for Docker API access
5. Use the `temp_workspace` fixture for temporary directories

Example:
```python
def test_my_docker_feature(docker_client, temp_workspace):
    """Test description."""
    container_name = "nodetool-test-myfeature"

    try:
        # Your test code
        container = docker_client.containers.run(...)
        # Assertions
    finally:
        # Cleanup
        try:
            container = docker_client.containers.get(container_name)
            container.remove(force=True)
        except docker.errors.NotFound:
            pass
```

## Best Practices

1. **Always clean up** - Use try/finally blocks to ensure containers are removed
2. **Use unique names** - Prefix test containers with `nodetool-test-`
3. **Use lightweight images** - Prefer `nginx:alpine` or similar small images
4. **Add timeouts** - Don't let containers run indefinitely
5. **Test isolation** - Each test should be independent
6. **Skip unavailable images** - Use `check_test_image_available()` and `pytest.skip()`

## Troubleshooting

### Tests Fail with "Docker is not available"
- Ensure Docker daemon is running: `docker ps`
- Check Docker installation: `docker version`

### Tests Fail with "Connection refused"
- Docker daemon might not be accessible
- Check Docker socket permissions
- Try: `sudo usermod -aG docker $USER` (then log out/in)

### Tests Leave Orphaned Containers
- List test containers: `docker ps -a | grep nodetool-test`
- Clean up manually: `docker rm -f $(docker ps -aq --filter name=nodetool-test)`

### Tests Are Slow
- Integration tests are intentionally slower than unit tests
- They test real Docker functionality
- Run unit tests during development: `pytest tests/deploy/test_self_hosted.py`
