"""Custom callback factories a spec's `runtime.callbacks` can name with
`{"kind": "custom", "params": {"path": "app.callbacks.<module>:<factory>",
"params": {...}}}` -- only when settings.allow_custom_callbacks is on, and
only under settings.custom_callback_module_prefixes (default:
"app.callbacks."). See app/factory/callback_registry.py for the contract and
app/callbacks/examples.py for working examples."""
