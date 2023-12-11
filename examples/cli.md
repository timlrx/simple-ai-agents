# CLI examples

## File Summary

```sh
cat README.md | aichat --model ollama/mistral "Summarize this file"
```

## Automatic Change Logs

```sh
git log 0.1.0..main --oneline | aichat --model ollama/mistral "Generate a change log for this release:"
```

## Shell assistant

```sh
aichat --model ollama/mistral "Use the given shell history:\n $(history | tail -10)\n to answer my subsequent questions."
```

## Website summaries

### Hackernews

Requires [ttok](https://github.com/simonw/ttok) and [strip-tags](https://github.com/simonw/strip-tags). Examples adapted from Simon Willison's [blog post](https://simonwillison.net/2023/May/18/cli-tools-for-llms/)!

```sh
curl -s https://news.ycombinator.com \
    | strip-tags \
    | ttok -t 4000 \
    | aichat --system 'summary bullet points'
```

### New York Times

```sh
curl -s https://www.nytimes.com \
  | strip-tags .story-wrapper \
  | ttok -t 4000 \
  | aichat --system 'summary bullet points'
```
