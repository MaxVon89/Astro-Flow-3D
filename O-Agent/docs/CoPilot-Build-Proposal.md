Based on the MD, coding the entire autonomous research-to-code system in one request could consume substantial credits. The proposed scope includes:

- LangGraph planning and retry workflow
- Research ingestion
- Scientific validation
- Experiment memory
- GPU job orchestration
- GitHub PR automation
- Tests, configuration, and documentation

The estimated 600–800 lines is manageable, but the surrounding research, integration, debugging, and validation are the expensive parts. You are unlikely to run out from a small first implementation, but a complete build in one uninterrupted session could be costly or hit usage limits.

A credit-conscious approach would be:

1. Build the structured project memory and planner.
2. Add a hardcoded local executor.
3. Add scientific validation and tests.
4. Integrate GPU submission only after the loop works.
5. Add GitHub PR automation last.

I can work incrementally and keep each step narrowly scoped, so you can stop after any phase and review the result.