---
name: Investigate
description: Scaffold a structured empirical research investigation with dated folders and documentation.
category: Research
tags: [investigation, research, scratch, exploration]
args: topic
---

**Goal**
Create a structured investigation folder and systematically explore a research question, capturing findings incrementally.

**Investigation Types**

| Type | When to Use | Focus |
|------|-------------|-------|
| **tracing** | "trace X to Y", "follow the data flow" | Call stacks, data flows, dependency chains |
| **archeology** | "document how X works", "map the system" | Comprehensive system documentation |
| **debugging** | "why is X failing", "diagnose the issue" | Reproduction, root cause, fix proposals |
| **feasibility** | "can we do X", "is it possible to" | Proof-of-concept, tradeoff analysis |
| **api** | "how does library X work", "investigate API" | External library API exploration |
| **design** | "compare options for X", "evaluate approaches" | Design questions, alternative comparison |

**Auto-Detection Logic**
- Contains "trace" or "follow" → tracing
- Contains "document how" or "map" → archeology
- Contains "debug" or "diagnose" or "failing" → debugging
- Contains "can we" or "possible" or "feasibility" → feasibility
- Contains "api" or "library" or "sleap-io" → api
- Contains "compare" or "evaluate" or "design" → design
- Default: feasibility (most open-ended)

**Steps**

1. **Create investigation folder**:
   - Path: `_scratch/YYYY-MM-DD-{topic-slug}/`
   - Use kebab-case for topic slug

2. **Generate README.md** with template based on type:
   ```markdown
   # Investigation: {title}

   **Date**: YYYY-MM-DD
   **Type**: {type}
   **Status**: active

   ## Task
   {Original question or problem statement}

   ## Background
   {Context, prior knowledge, constraints - fill in during investigation}

   ## Checklist
   - [ ] {Initial steps based on investigation type}

   ## Quick Findings
   {Bullet points updated as discoveries are made}

   ## Code Examples
   ```python
   # Working code snippets discovered during investigation
   ```

   ## Related
   - {Links to relevant files, external docs}
   ```

3. **Create findings.md** for detailed notes:
   ```markdown
   # Findings: {title}

   ## Session Log

   ### YYYY-MM-DD
   {Detailed findings from today's exploration}
   ```

4. **Create subdirectories as needed**:
   - `scripts/` - Analysis scripts (Python, shell, etc.)
   - `notebooks/` - Jupyter notebooks
   - `data/` - Intermediate data files

5. **Begin investigation interactively**:
   - Populate initial checklist based on investigation type
   - Start exploring, updating README.md as you go
   - Use Quick Findings for incremental capture
   - Test code examples in scripts/ directory

**Type-Specific Checklists**

For **tracing**:
- [ ] Identify start and end points
- [ ] Trace forward path (source → destination)
- [ ] Document each transformation step
- [ ] Note coupling points and dependencies

For **archeology**:
- [ ] Identify entry points
- [ ] Map layer 1: top-level components
- [ ] Map layer 2: key subsystems
- [ ] Document configuration options
- [ ] Create architecture diagram

For **debugging**:
- [ ] Document reproduction steps
- [ ] Identify error/symptom location
- [ ] Trace backward to root cause
- [ ] Identify fix candidates
- [ ] Document workarounds

For **feasibility**:
- [ ] Define success criteria
- [ ] Create minimal proof-of-concept
- [ ] Document what works
- [ ] Document what doesn't work
- [ ] Assess effort vs. value

For **api**:
- [ ] Read library documentation
- [ ] Find relevant classes/functions
- [ ] Write minimal test scripts
- [ ] Document API patterns
- [ ] Create working examples
- [ ] Identify integration points with our code

For **design**:
- [ ] List design questions (DQ1, DQ2, ...)
- [ ] Identify alternatives for each
- [ ] Document tradeoffs
- [ ] Make recommendations
- [ ] Note open questions

**Parallel Exploration**

For complex investigations with independent aspects, spawn parallel agents:

```
Investigation: How does sleap-io handle video loading?
├── Agent 1: Trace video loading pipeline
├── Agent 2: Document ImageVideo vs HDF5Video backends
└── Agent 3: Test path remapping options
```

Track spawned agents in the README and consolidate findings.

**Completion**

When investigation is complete:
1. Mark all checklist items done
2. Update status to "completed"
3. Summarize key findings in README
4. Create code examples that can be reused
5. If findings affect the codebase, create follow-up tasks

**Important Notes**

- `_scratch/` is gitignored - files are never version controlled
- Never delete scratch files - they may be needed later
- Distill findings into code or documentation in the main codebase
- Keep investigations focused - split into sub-investigations if scope grows
- Use `uv run python script.py` to run test scripts

**Usage**
```
/investigate how does sleap-io load videos from image directories
/investigate --type debugging why is Series.video returning None
/investigate --type api sleap-io video backend options
/investigate --type design compare options for image path remapping
```