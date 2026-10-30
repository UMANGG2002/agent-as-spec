"""Conversational sessions with compiled spec agents, built for horizontal
scale:

    client --HTTP--> API (stateless) --> SessionManager
                                            |  enqueue (Redis Streams work queue, sharded)
                                            v
                                   runner workers (N replicas)
                                            |  ADK Runner(App(compiled spec agent,
                                            |      plugins, compaction, resumability),
                                            |      DatabaseSessionService, mem0 memory)
                                            v
                         per-invocation Redis event stream  --SSE (Last-Event-ID)--> client

See docs/PRODUCTION_ARCHITECTURE.md for the full design.
"""
