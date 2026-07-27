# Changes

## English package integration

- Converted user-facing CLI and documentation to English.
- Reorganized implementation modules into the installable `perception_tools`
  package.
- Added `perception-tools` and `perception-tools-mcp` console commands.
- Preserved all 53 registered tools across five categories.
- Made optional integrations lazy so missing extras do not break imports.
- Standardized hosted model calls on the official OpenAI API.
- Replaced stale paths, setup instructions, and tool counts.
- Made live external-service tests opt-in.
- Added package, CLI, server, language, dependency, and provider regression
  tests.
