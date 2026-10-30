"""Production platform primitives shared by every module: request context,
structured logging, Redis, caching, rate limiting, resilience, outbound HTTP
safety, and the transactional outbox. Nothing in here knows about
capabilities, specs or agents -- those modules build on top of it.
"""
