---
name: help-me-plan
description: Interviews the user one question at a time with AskUserQuestion about a feature or change in the current codebase, pushes back on their answers and on gaps between intent and existing code, then offers to draft a plan in plan mode. Use only when the user types /help-me-plan.
disable-model-invocation: true
argument-hint: "[what you want to plan]"
---

# Help Me Plan

This skill runs a grilling style interview about something the user wants to build or change in this codebase. The goal is shared understanding, and the user's own learning, before any plan or code exists. The user owns every decision. You own finding facts, giving recommendations, and pushing back.

## Rules that always apply

1. Ask exactly one question per AskUserQuestion call. The questions array must contain one item, never more.
2. Every question has two to four options. Put your recommended option first and add "(Recommended)" to its label. Give every option a short description that states its main trade off.
3. AskUserQuestion always adds an "Other" box. Remind the user once, at the start, that they can type their own answer there for any question.
4. Never answer a decision on the user's behalf. Decisions wait for the user. Facts are your job: if the codebase, git history, or a quick read can answer something, find it yourself instead of asking.
5. Do not enter plan mode, edit files, or write code during the interview. Plan mode comes only at the end, and only if the user says yes.
6. Use plain, simple language in full sentences.

## Step 1: Choose the topic

If the user gave a topic with the command ($ARGUMENTS is not empty), restate it in one sentence and go to Step 2.

If no topic was given, look at the current branch, `git status`, recent commits, and TODO or FIXME comments. Then use AskUserQuestion to offer two to four likely topics based on what you found, with the most likely one recommended. The user can type a different topic in the "Other" box.

## Step 2: Explore the codebase

Before the first question, read the parts of the codebase that relate to the topic. For broad searches, use the Explore subagent so the main conversation stays focused. Learn what already exists, the conventions in use, and where the topic touches existing code.

Then send the user a short summary of three to five sentences describing what you found, with file paths. This grounds the interview in the real code.

## Step 3: Build the decision tree

Privately list every decision the topic involves, and note which decisions depend on others. Sort them into two groups:

1. Facts, which you can settle by reading code or data. Settle these yourself.
2. Decisions, which only the user can make. These become questions.

Ask decisions in dependency order. Only ask a question when every decision it depends on is already settled. Never ask something whose answer could change because of a question you have not asked yet.

## Step 4: Run the interview

For each question, the question text should give one or two sentences of context and say why the decision matters. When comparing code shapes, use the option preview field to show short snippets.

### Push back after every answer

After each answer, check the following before moving on:

1. **Intent versus code.** Does the answer conflict with what is already implemented? If so, cite the file path and explain the conflict, then ask whether the code or the intent should change.
2. **Consistency.** Does the answer contradict an earlier answer? If so, point it out and ask which one wins.
3. **Missing cases.** Has the user missed an edge case or failure mode? Think about data quality, scale, latency, error handling, testing, monitoring, rollout, and for ML work, data leakage and the gap between offline and online metrics. Raise the most important one as the next question.
4. **Weak reasoning.** If a choice seems weak or unexplained, ask why before moving on.

If a later answer changes an earlier decision, reopen that branch and ask about it again.

### Protect the user's learning

1. **Open questions.** If a question has no meaningful set of options, such as "What problem does this solve?", ask it in plain text instead of using AskUserQuestion.
2. **Passive answering.** If the user picks the recommended option three times in a row, ask them in plain text to explain, in their own words, the reasoning behind one of those choices. Then continue.
3. **"I don't know".** Treat this as a real answer. Mark the decision as open and suggest how the user could find out.
4. **Questions that talk cannot settle.** Some questions need something to look at first, such as how results look on real data. Say so, suggest a quick prototype, notebook check, or query, and mark the decision as open instead of rephrasing the question.

### Keep the user oriented

About every five questions, send a short progress note: what is settled, and roughly how many branches remain. The user can type "done" or "wrap up" at any time to move to Step 5.

## Step 5: Confirm the decision summary

When every branch is visited, or the user asks to wrap up, write a decision summary in the conversation. Do not save it to a file. Use these sections:

1. **Goal.** One or two sentences on what is being built and why.
2. **Decisions.** Each decision with its short reason, using the user's own words where possible.
3. **Existing code versus changes.** What already exists, and what will change, with file paths.
4. **Open questions.** Anything marked open, with the suggested way to resolve it.
5. **Out of scope.** Anything the user chose not to do.

Then use AskUserQuestion to ask whether the summary is correct, with the options "Yes, it is correct (Recommended)" and "I want to change something". If the user wants changes, make them and confirm again.

## Step 6: Offer to create a plan

Use AskUserQuestion to ask: "Do you want me to create an implementation plan now?" Offer "Yes, create the plan (Recommended)" and "No, stop here".

If the user says no, end the session. The confirmed summary is the record.

If the user says yes:

1. Finish any remaining codebase reading first, because plan mode limits what you can do.
2. Call the EnterPlanMode tool. If it is deferred, load it with ToolSearch first. The user will be asked to approve the switch. If the tool is not available, ask the user to press Shift+Tab twice to enter plan mode.
3. Write the plan using only the confirmed decision summary. Every step must trace back to a confirmed decision. Do not add new decisions. If you find a gap, ask about it with AskUserQuestion before continuing.
4. Structure the plan with these sections: Goal, Relevant Files, Steps, Testing and Validation, Open Questions, and Out of Scope.
5. Call ExitPlanMode to present the plan for the user's approval.
