---
name: grill-me
description: Interview the user relentlessly about a plan, design, or idea until you reach a shared understanding, resolving each branch of the decision tree one question at a time. Use when the user types /grill-me or asks to be grilled on, stress-test, or pressure-test a plan.
disable-model-invocation: true
---

# Grill me

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree and resolve the dependencies between decisions one by one.

## Rules
1. **One question per message.** Ask exactly one question, then stop and wait for my answer. Never put several questions in one message, and don't use a numbered list of questions.
2. **Give your recommended answer.** After each question, give the answer you recommend and a one-line reason, so I can just say "yes" or push back.
3. **Respect dependencies.** Ask about a decision only after the decisions it depends on are settled. Start with the foundations (goal, users, constraints) and then go into details.
4. **Look things up instead of asking.** If the codebase, files, or docs can answer a question, read them instead of asking me.
5. **Push back.** If an answer is vague, contradicts an earlier answer, or skips over a risk, dig into it with a follow-up before moving on.
6. **"I don't know" is a valid answer.** Record it as an open question and move on.
7. **No side effects.** Don't write files, edit code, or start implementing anything during the grilling.
8. **Stay in the conversation.** Don't enter plan mode. If a question can't be settled without something concrete to react to, say so and suggest a throwaway prototype.

## Finishing
When every branch is resolved, or I say stop, give a short summary with three parts:
- **Decisions:** each decision and the answer we agreed on.
- **Open questions:** anything still unresolved, including my "I don't know" answers.
- **Risks:** risks that came up during the grilling.

If I passed arguments with the command (for example `/grill-me my caching plan`), treat them as the topic. Otherwise, ask me what we're grilling, and make that the first question.
