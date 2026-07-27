# Tool Reference

The installed CLI is the source of truth for all 53 tools and their signatures:

```bash
perception-tools list
perception-tools list --category search
perception-tools list --category multimodal
perception-tools list --category filesystem
perception-tools list --category public
perception-tools list --category private
perception-tools info TOOL_NAME
```

The catalog contains 4 search, 19 multimodal, 3 filesystem, 25 public-data,
and 2 private-data tools.

Call a tool with `key=value` arguments:

```bash
perception-tools run grep pattern=async directory=perception_tools file_pattern='*.py'
perception-tools run weather location=Boston
perception-tools run currency_converter amount=100 from_currency=USD to_currency=EUR
```

Values are parsed as JSON when possible, so booleans, numbers, arrays, and
objects can be passed without custom syntax. Use `perception-tools info` to see
required parameters, defaults, network requirements, and credential notes.
