# Security — Secrets, API Key Rotation, Audit Logs, Guardrails

> Eliminate secret sprawl with centralized vaults (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Never put credentials in config files, env files in VCS, or spreadsheets. Use IAM roles instead of static keys; use OIDC for CI/CD. The AI gateway pattern is the 2026 solution: app -> gateway -> model provider, with the gateway pulling credentials from the vault at runtime. Rotate in the vault and all apps pick up the new key within minutes — no redeployment, no asking "who has the new key" in Slack. Rotation policy <= 90 days; scan every commit with TruffleHog / GitGuardian / Gitleaks. Zero trust: MFA, SSO, RBAC/ABAC, short-lived tokens, device posture. PII scrubbing uses entity recognition to mask PHI/PII before forwarding; consistent tokenization (Mesh approach) maps sensitive values to stable placeholders so the LLM preserves code/relationship semantics. Network egress: LLM services in a dedicated VPC/VNet subnet, allow-list only `api.openai.com`, `api.anthropic.com`, etc.; block all other outbound. 2026 incident cause: Vercel supply-chain attack via compromised CI/CD credentials exfiltrated env vars across thousands of customer deployments.

**Type:** Learn
**Languages:** Python (standard library, a toy PII scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateway), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- List four secrets-management anti-patterns (config files in VCS, hardcoded env, spreadsheets, static keys) and name their replacements.
- Explain the "AI gateway pulls from vault" pattern as the 2026 production standard.
- Implement a PII scrubber with consistent tokenization (same value -> same placeholder) so semantics survive.
- Cite the 2026 Vercel supply-chain incident and its lesson for CI/CD credential hygiene.

## The Problem

An intern commits a `.env` with an API key. They quickly delete it. The key is already in git history — GitGuardian scanning catches it, and your rotation process is "notify the team in Slack, update 40 config files, redeploy all services." 8 hours later, half the services are up and the other half are waiting for deploy windows.

Separately, a user prompt contains "My SSN is 123-45-6789." The prompt is sent to OpenAI. You have a BAA, but your internal policy is to mask PII before forwarding. You didn't.

Separately, your EKS cluster's LLM pods can reach any internet host. Someone exfiltrates data by making DNS queries to an attacker-controlled domain. Nothing stops it.

LLM service security must address all three vectors. Vault-backed credentials. PII scrubbing. Network egress filtering. Audit logs.

## The Concept

### Centralized Vault + IAM Role Pull

**Vault**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. Single source of truth.

**IAM role**: applications/gateways authenticate with their IAM identity, no static keys. Vault returns secrets for the token lifetime.

**AI gateway pattern**: the gateway pulls `OPENAI_API_KEY` from the vault at request time. Rotate in the vault; the next request picks up the new key. No redeployment needed.

### Rotation Policy <= 90 Days

All API keys, vault root tokens, CI/CD credentials. Automate rotation where possible. Manual rotations must be documented and tracked.

### Secret Scanning

- **TruffleHog** — regex + entropy on commits.
- **GitGuardian** — commercial, high accuracy.
- **Gitleaks** — OSS, runs in CI.

Run on every commit. Block the PR when a new secret is detected.

### Zero-Trust Posture

- MFA required on all accounts.
- SSO via SAML/OIDC.
- Fine-grained access with RBAC (role-based) or ABAC (attribute-based).
- Short-lived tokens (hours, not days).
- Device posture — only corporate devices with disk encryption.

### PII / PHI Scrubbing

Before a prompt leaves your infrastructure:

1. Entity recognition (spaCy NER, Presidio, commercial).
2. Mask matched entities: `"My SSN is 123-45-6789"` -> `"My SSN is [SSN_TOKEN_A3F]"`.
3. Consistent tokenization (Mesh approach): same value maps to same placeholder, so the LLM preserves relationships.
4. Optional reverse-mapping on the LLM response.

Static regex filters catch basic patterns; NER catches more. Use both.

### Input + Output Guardrails

Input: block known jailbreaks, prohibited topics; rate-limit per user.

Output: regex-scrub leaked secrets (API key patterns, email patterns in refusal contexts), classify policy violations with a classifier.

### Network Egress Allow-List

LLM services in a dedicated subnet:
- Allow-list: `api.openai.com`, `api.anthropic.com`, vector DB endpoints, vault endpoints.
- Everything else: drop.
- DNS goes through an allow-list-only resolver (prevents DNS tunnel exfiltration).

### Audit Logs

Immutable log of every LLM call, with:
- Timestamp.
- User / tenant.
- Prompt hash (don't store raw prompts for privacy).
- Model + version.
- Token count.
- Cost.
- Response hash.
- Any guardrail triggers.

Retain per regulatory requirements (SOC 2: one year, HIPAA: six years).

### The 2026 Vercel Incident

Supply-chain attack: compromised CI/CD credentials exfiltrated env vars across thousands of customer deployments. Lesson: CI/CD credentials are production credentials. Store them in a vault. Scope them narrowly. Rotate aggressively.

### Numbers You Should Remember

- Rotation policy: <= 90 days.
- Scan every commit: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: CI/CD credentials compromised -> thousands of customer env vars leaked.
- Audit log retention: SOC 2 = one year, HIPAA = six years.

## Use It

`code/main.py` implements a toy PII scrubber with consistent tokenization and an append-only audit log.

## Ship It

This lesson produces `outputs/skill-llm-security-plan.md`. Given regulatory scope and current state, it plans vault migration, scrubber, egress, and audit logs.

## Exercises

1. Run `code/main.py`. Send two prompts referencing the same SSN. Confirm both get the same placeholder.
2. Design a network egress policy for a vLLM-on-EKS deployment calling OpenAI + Anthropic + Weaviate.
3. You find a key in git history (from two years ago). What's the correct response — rotate the key, scrub history, or both? Argue.
4. Your audit logs grow 10 GB/day. Design retention tiers (hot 30 days, warm 12 months, cold 6 years).
5. Argue whether reverse-tokenization (replacing real values back into LLM responses) is worth the complexity versus keeping placeholders visible.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Vault | "Secret store" | Centralized credential management service |
| IAM role | "Identity-based auth" | Role assumed by apps; returns short-lived credentials |
| OIDC for CI/CD | "Cloud-issued token" | No static keys in CI — identity via OIDC |
| TruffleHog / GitGuardian / Gitleaks | "Secret scanners" | Commit-time secret detection |
| RBAC / ABAC | "Access control" | Role-based vs attribute-based |
| PII scrubbing | "Data masking" | Remove or tokenize sensitive entities |
| Consistent tokenization | "Stable placeholders" | Same value -> same token every time |
| Mesh approach | "Mesh tokenization" | Semantic-preserving tokenization pattern |
| Egress allow-list | "Outbound allow-list" | Only permitted domains are reachable |
| Audit log | "Immutable history" | Append-only record for compliance |

## Further Reading

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
