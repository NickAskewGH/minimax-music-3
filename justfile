# List available tasks.
default:
    @just --list

# Install dependencies from the lockfile.
sync:
    uv sync --locked --extra inference

# Refresh the dependency lockfile.
lock:
    uv lock

# Start the local browser studio.
studio:
    uv run --extra inference music3 serve

# Check weights and GPU access.
doctor:
    uv run --extra inference music3 doctor

# Run the CLI (for example: just music generate --help).
music *args:
    uv run --extra inference music3 {{args}}

# Run the test suite without synchronizing away installed inference dependencies.
test:
    uv run --extra inference pytest
