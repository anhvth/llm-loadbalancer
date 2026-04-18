# Changelog

## Unreleased

- Added `endpoints.setup` with `ssh` and `direct` modes so the load balancer can route directly to reachable worker hosts without SSH tunnels.
- Updated the default and example configuration for the direct worker setup used on the current cluster.
- Hardened startup by waiting for healthy upstreams before failing.
- Scaled the Uvicorn listen backlog with total configured concurrency to handle larger connection bursts more reliably.
- Updated tests and documentation for the new direct-mode and high-concurrency behavior.
