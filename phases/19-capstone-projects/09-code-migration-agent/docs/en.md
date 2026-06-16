# Capstone Project 09 — Code Migration Agent (Repo-Level Language / Runtime Upgrade)

> Amazon's MigrationBench (Java 8 to 17) and Google's App Engine Py2-to-Py3 migrator set the 2026 standard. Moderne's OpenRewrite performs deterministic AST rewrites at scale. Grit targets the same problem with a codemod-style DSL. The production paradigm combines both: a deterministic substrate for safe rewrites, an agent layer for ambiguous cases, a sandbox for per-branch builds, and a test harness that goes green before a PR is opened. This capstone project migrates 50 real repositories and publishes a pass rate with a failure taxonomy.

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (Transformers), Phase 11 (LLM Engineering), Phase 13 (Tools), Phase 14 (Agents), Phase 15 (Autonomous Systems), Phase 17 (Infrastructure)
**Phases involved:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## The Problem

Large-scale code migration is one of the cleanest production applications of coding agents in 2026. Ground truth is obvious (does the test suite pass after migration?), payoff is real (migrating a Java-8 fleet is a headcount-year project), and benchmarks are public (MigrationBench's 50-repo subset). Moderne's OpenRewrite handles the deterministic side. The agent layer handles everything OpenRewrite recipes cannot: ambiguous rewrites, build-system drift, long-tail syntax, and transitive dependency breakage.

You will build an agent that takes a Java 8 repository (or Python 2 repository), produces a CI-green migration branch. You will measure pass rate, test coverage retention, cost per repository, and build a failure taxonomy. The side-by-side comparison against the purely deterministic baseline shows where the agent's value actually lies.

## The Concept

The pipeline has two layers. **Deterministic substrate** (OpenRewrite for Java, libcst for Python) safely handles the bulk of mechanical rewrites: imports, method signatures, null-safety changes, try-with-resources, deprecated API replacements. It is fast and produces auditable diffs. **Agent layer** (OpenAI Agents SDK or LangGraph built on Claude Opus 4.7 and GPT-5.4-Codex) handles the cases recipes cannot: build file upgrades (Maven/Gradle/pyproject), transitive dependency conflicts, test flakiness, and custom annotations.

Each repository gets a Daytona sandbox preloaded with the target runtime. The agent iterates: run the build, classify failures, apply a fix, rerun. Hard limits: 30 minutes per repo, $8 per repo, 20 agent turns. If all tests pass and coverage delta is non-negative, the branch opens a PR. If not, the repository is filed under a failure class with evidence.

The failure taxonomy is the deliverable. Across 50 repos, what broke? Transitive dependencies? Custom annotations? Build tool versions? Test flakiness unrelated to migration? Each category gets a count and an example diff. Future recipe authors can target the top three.

## Architecture

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Tech Stack

- Deterministic substrate: OpenRewrite (Java) or libcst (Python)
- Agent: OpenAI Agents SDK or LangGraph built on Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainer per branch, preloaded with target runtime (Java 17 / Python 3.12)
- Build systems: Maven, Gradle, uv (Python)
- Benchmark: Amazon MigrationBench 50-repo subset (Java 8 to 17), Google App Engine Py2-to-Py3 repos
- Test harness: Parallel runner, coverage via Jacoco (Java) or coverage.py (Python)
- Observability: Langfuse + per-repo trace bundle with per-hunk diffs
- Dashboard: Failure taxonomy dashboard with per-category counts and example diffs

## Build It

1. **Recipe pass.** Run OpenRewrite (Java) or libcst (Python) recipes first. Consume 70-80% of mechanical migration. Commit as a "recipe" commit.

2. **Build trial.** Daytona sandbox: install target runtime, run build. Green? Skip to tests. Red? Hand off to the agent.

3. **Agent loop.** LangGraph with tools: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. The agent classifies failures (dependency, syntax, test, build tool) and applies a targeted fix. Rerun.

4. **Budget cap.** 30-minute wall clock, $8 cost, 20 agent turns per repo. Any exceeded limit halts and files under "budget_exhausted" with the current diff.

5. **Test + coverage gate.** After the build goes green, run the test suite. Compare coverage against the baseline repo. If coverage drops more than 2%, file under "coverage_regression."

6. **Open PR.** On success, push branch, open PR with diff and a summary explaining which recipes were applied and which commits the agent wrote.

7. **Failure taxonomy.** For each failed repo, tag a category: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. Build a dashboard.

8. **50-repo batch run.** Execute on the MigrationBench subset. Report pass rate per category, cost per repo, coverage retention, and a comparison against the purely deterministic baseline.

## Use It

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Ship It

`outputs/skill-migration-agent.md` is the deliverable. Given a repository, it runs deterministic recipes then an agent loop, producing a green migration branch or filing the repo under a taxonomy category.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | MigrationBench pass rate | pass@1 on 50-repo subset |
| 20 | Test coverage retention | Average coverage delta vs baseline |
| 20 | Cost per migrated repo | $/repo on the passing batch |
| 20 | Agent / deterministic tool integration | Fraction of fixes handled by OpenRewrite vs agent-written |
| 15 | Failure analysis write-up | Taxonomy completeness with examples |
| **100** | | |

## Exercises

1. Run the migration pipeline with OpenRewrite only (no agent). Compare pass rate to the full pipeline. Identify the cases where only the agent makes a difference.

2. Implement a "lint-clean" check: after migration, run a style linter (spotless for Java, ruff for Python). Fail the PR if new lint errors appear. Measure the rate of "coverage retained but style regressed."

3. Add a "minimal-diff" optimizer: after the agent's branch passes tests, use a second pass to trim unnecessary changes. Report the diff-size reduction.

4. Extend to a third migration: Node 18 to Node 22. Reuse the sandbox wrapper; swap the recipe layer for a custom codemod.

5. Measure "time to first green build" (TTFGB) as an experience metric. Target: p50 under 10 minutes.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Deterministic substrate | "recipe engine" | OpenRewrite / libcst: declarative AST rewrites with safety guarantees |
| Codemod | "code-changing program" | A rewrite rule that mechanically modifies source code |
| Build drift | "tool version skew" | Subtle behavior changes in Maven / Gradle / uv across major versions |
| Failure class | "taxonomy bucket" | An annotated reason why a repo did not migrate: dependency, syntax, test, build tool, budget |
| Coverage delta | "coverage retention" | The percentage change in test coverage from baseline to migrated branch |
| Agent turn | "tool-call round" | One plan -> act -> observe cycle in the agent loop |
| Budget exhaustion | "hit the cap" | A repo exhausted its 30-minute / $8 / 20-turn budget without passing |

## Further Reading

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — the 2026 standard benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — deterministic substrate reference
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — alternative codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK reference
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — alternative migration benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python deterministic substrate
- [Daytona sandboxes](https://daytona.io) — reference per-branch sandbox
