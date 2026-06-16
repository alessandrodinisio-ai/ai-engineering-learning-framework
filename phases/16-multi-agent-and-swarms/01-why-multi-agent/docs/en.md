# Why Multi-Agent?

> A single agent hits walls. The smart move isn't building a bigger agent — it's using more agents.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 min

## Learning Objectives

- Identify single-agent ceilings (context overflow, expertise blurring, serial bottlenecks) and articulate when splitting into multiple agents is the right call
- Compare orchestration patterns (pipeline, parallel fan-out, supervisor, hierarchical) and choose the right one for a given task structure
- Design a multi-agent system with clear role boundaries, shared state, and communication contracts
- Analyze multi-agent complexity tradeoffs (latency, cost, debugging difficulty) versus single-agent simplicity

## The Problem

You built a single agent in Phase 14. It works. It reads files, runs commands, calls APIs, reasons about results. Then you throw it at a real codebase: 200 files, three languages, infrastructure-dependent tests, plus a requirement to "research an external API before writing code."

The agent stalls. Not because the LLM is dumb — the task exceeds what a single agent loop can handle. The context window fills with file contents. The agent forgets what it read 40 tool calls ago. It tries to be researcher, programmer, and reviewer simultaneously, and does all three poorly.

This is the single-agent ceiling. You hit it every time a task requires any of the following:

- **Context exceeds a single window** — reading 50 files blows past 200k tokens
- **Different phases need different expertise** — the prompt for research is nothing like the prompt for code generation
- **Some work can be parallelized** — why read three files sequentially when you could read them simultaneously?

## The Concept

### The Single-Agent Ceiling

A single agent is one loop, one context window, one system prompt. Picture this:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

Three things break:

1. **Context saturation** — tool results pile up. By turn 30, the agent has consumed 150k tokens of file contents, command outputs, and prior reasoning. Critical details from turn 5 get flushed.

2. **Role confusion** — a system prompt that says "you are researcher, programmer, reviewer, and tester" produces an agent that half-researches, half-codes, and never finishes a review.

3. **Serial bottleneck** — the agent reads file A, then B, then C. Three serial LLM calls, three serial tool executions. Zero parallelism.

### The Multi-Agent Solution

Split the work. Give each agent one job, one context window, one system prompt tuned for that job:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

Each agent has:
- A focused system prompt ("You are a code reviewer. Your only job is finding bugs.")
- Its own context window (not polluted by other agents' work)
- Clear input/output contracts (receives research notes, outputs code)

### Systems That Actually Do This

**Claude Code subagents** — when Claude Code spawns a subagent via `Task`, it creates a child agent with a constrained task. The parent keeps its own context clean. The child does focused work and returns a summary.

**Devin** — runs a planner agent, a coder agent, and a browser agent. The planner decomposes into steps, the coder writes code, the browser researches documentation. Each has separate context.

**Multi-agent coding teams (SWE-bench)** — the best-performing systems on SWE-bench use a researcher that reads the codebase, a planner that designs the fix, and a coder that implements it. Single-agent systems score lower.

**ChatGPT Deep Research** — spawns multiple search agents in parallel, each exploring a different angle, then synthesizes results.

### The Spectrum

Multi-agent isn't binary — it's a continuous spectrum:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** — one loop, one prompt. Good for simple tasks.

**Subagents** — a parent agent spawns child agents for focused subtasks. The parent maintains the overall plan; children report back. This is what Claude Code does.

**Pipeline** — agents execute sequentially. Agent A's output becomes Agent B's input. Good for staged workflows: research -> code -> review -> test.

**Team** — agents run in parallel, sharing a message bus. Each has its own role, coordinated by an orchestrator. Good for tasks requiring multiple skills simultaneously.

**Swarm** — many identical or near-identical agents, shared state. No fixed orchestrator. Agents pick up work from a queue. Good for high-throughput parallel tasks.

### Four Multi-Agent Patterns

#### Pattern 1: Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Each agent transforms data and passes it downstream. Easy to reason about. A failure in one stage blocks all subsequent stages.

#### Pattern 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Split work across parallel agents, then merge results. Good for tasks that decompose into independent subtasks.

#### Pattern 3: Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

A smart orchestrator decides what to do, delegates work to workers, and synthesizes results. The orchestrator itself is an agent whose tools include spawning and managing other agents.

#### Pattern 4: Peer Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

No central orchestrator. Agents communicate peer-to-peer. Decisions emerge from interactions. Harder to debug, but scales to many agents.

### When NOT to Use Multi-Agent

Multi-agent introduces complexity. Every message between agents is a potential failure point. Debugging goes from "read one conversation" to "trace messages across five agents."

**Stay with a single agent when:**
- The task fits in one context window (working data under ~100k tokens)
- You don't need different system prompts for different phases
- Serial execution is fast enough
- The task is simple enough that splitting it adds more overhead than value

**Complexity costs:**
- Every agent boundary is a lossy compression: Agent A's full context gets compressed into a message sent to Agent B
- Coordination logic (who does what, when, in what order) is itself a source of bugs
- Latency goes up: N agents means at minimum N serial LLM calls, more if they need back-and-forth
- Cost multiplies: each agent burns tokens independently

Rule of thumb: if a task takes fewer than 20 tool calls and fits in 100k tokens, keep it single-agent.

## Build It

### Step 1: The Overloaded Single Agent

Here's a single agent trying to do everything. It has one massive system prompt and one context window holding research, code, and review simultaneously:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Problems with this approach:
- The context window grows with every phase. By the review step, it holds research notes, code, and prior reasoning all at once.
- The system prompt is generic — it can't be tuned for each phase.
- Nothing runs in parallel.

### Step 2: Specialist Agents

Now split it up. Each agent does one job:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

Each specialist has a focused prompt. Each gets a clean context window with only its required input.

### Step 3: Coordination via Messages

Wire the specialists together with explicit message passing:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Each agent receives only messages addressed to it. No context pollution. The researcher's 50k-token documentation reading never enters the reviewer's context.

### Step 4: Comparison

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

The multi-agent version uses more total tokens (three agents, three independent LLM calls), but each agent's context stays clean. Quality improves at each stage because system prompts are specialized.

## Use It

This lesson produces a reusable prompt for deciding when to go multi-agent. See `outputs/prompt-multi-agent-decision.md`.

## Exercises

1. Add a fourth specialist agent: a "tester" that receives the coder's code and the reviewer's feedback, then writes tests
2. Modify the pipeline so the reviewer can send feedback back to the coder for one revision round (max 2 cycles)
3. Convert the serial pipeline to fan-out: run the researcher and a "requirements analyst" agent in parallel, merge their outputs, then pass to the coder

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "Hive mind for AI agents" | A group of peer agents sharing state with no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and dispatches but may not do work itself. |
| Coordinator | "Traffic cop" | A non-agent component (usually just code, not an LLM) that routes messages between agents by rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when output conflicts need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns arising from agent interactions that weren't explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Split a task across parallel agents (fan-out), then merge their results (fan-in). |
| Message passing | "Agents talking to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## Further Reading

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) — survey of multi-agent patterns
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) — Microsoft's multi-agent conversation framework
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) — how Claude Code uses Task for delegation
- [CrewAI documentation](https://docs.crewai.com/) — role-based multi-agent framework
