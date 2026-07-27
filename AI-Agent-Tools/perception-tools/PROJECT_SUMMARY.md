# Project Summary

Perception Tools provides 53 MCP-compatible tools for AI-agent perception and
retrieval. The project is packaged under `perception_tools`, installs console
entry points, and can be used from the CLI or any stdio MCP client.

The five tool groups cover search, multimodal content, local files, public data,
and credentialed private data. Core imports do not require every integration;
documents, media, data providers, and private services are installable extras.

The current version uses English throughout its code-facing interface and
documentation. Hosted model calls use only the official OpenAI API. Tests are
deterministic by default, with external-service coverage explicitly opt-in.
