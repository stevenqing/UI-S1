# Mind2Web Closed-API Baselines Preflight

Status: `BLOCKED_NO_API_CREDENTIALS_OR_VERSION_CONTRACT`

## Affected Rows

- GPT-4V + OmniParser.
- SeeAct closed-model planning/grounding configurations.
- GPT-4o + UGround, Aria-UI, or Aguvis.

## Current Environment

`OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`GOOGLE_API_KEY` are all unset. No secret values were read or logged.

## Reproduction Requirements

A valid run requires more than an API key:

1. the exact dated closed model or deployment revision used by the paper;
2. provider endpoint and image-processing behavior;
3. prompt, history, and action converter;
4. fixed detector/grounder checkpoint revisions where applicable;
5. request count, token/image cost budget, retry policy, and immutable raw
   response logging;
6. permission to transmit the protected benchmark screenshots to the provider.

Without these, running a current API alias would be a new experiment rather
than a paper baseline reproduction. No result is reported until the access,
version, privacy, and budget contract is supplied.
