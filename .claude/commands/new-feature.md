---
name: New Feature
description: End-to-end workflow for scoping, proposing, reviewing, and implementing a new feature using OpenSpec and TDD.
category: Development
tags: [feature, openspec, tdd, workflow]
args: feature_request
---

You are a scientific programmer that values testing, code quality, reproducibility, metadata preservation, traceability, interpretability, and performance. You are starting a new feature workflow. The user's feature request is: $ARGUMENTS

**Guardrails**

- Do NOT write any implementation code until the proposal is approved.
- Follow OpenSpec conventions strictly (see `openspec/AGENTS.md`).
- Use TDD when implementing (tests before implementation code).
- Always ask clarifying questions before proceeding if anything is vague, ambiguous, or underspecified. Do not assume.

**Steps**

1. **Ensure feature branch**: Check if you are on a feature branch (not `main`). If on `main`, ask the user what branch name to create (suggest one based on the feature), then create and switch to it before proceeding.

2. **Understand scope**: Use subagents to explore the codebase and understand the current state relevant to this feature. Investigate:
   - Existing trait modules in `sleap_roots/` (lengths, angles, tips, bases, etc.)
   - Pipeline classes and the networkx trait dependency graph in `sleap_roots/trait_pipelines.py`
   - Test fixtures in `tests/fixtures/` and test data in `tests/data/`
   - Related specs in `openspec/specs/`
   - Active changes in `openspec/changes/`

3. **Ask clarifying questions**: Based on what you learned from the codebase exploration, ask the user any clarifying questions about:
   - Requirements and expected behavior
   - Edge cases (empty arrays, NaN, single points)
   - Biological validity and scientific accuracy
   - Data handling and coordinate systems
   - Impact on published results or reproducibility
   - Memory considerations for large datasets
   - Which plant types are affected (dicot, monocot, etc.)
   - Which test data in `tests/data/` is relevant
   Do not proceed until you have clear answers.

4. **Create OpenSpec proposal**: Run `/openspec:proposal` to scaffold the change proposal, following all OpenSpec best practices. Ground the proposal in what you learned from steps 2-3. The proposal's `tasks.md` must explicitly outline a TDD approach: for each task, specify what tests will be written first and what behavior they verify before implementation begins.

5. **Review the proposal**: Run the openspec-review skill to have the proposal critically reviewed by 5 specialized subagents. If the review verdict is BLOCKED, fix the issues raised and re-run the review. Repeat until the verdict is APPROVED or NEEDS REVISION.

6. **Get user approval**: Present the reviewed proposal to the user and wait for explicit approval before proceeding to implementation.

7. **Implement with TDD**: Once approved, run `/openspec:apply` to implement the change using test-driven development. Write tests before implementation code.