---
name: help-me-brainstorm
description: Helps the user brainstorm on an open question, such as how to improve ad creative performance. Frames the problem briefly, offers grouped ideas, then scores and narrows them to one or two choices. Use only when the user types /help-me-brainstorm.
disable-model-invocation: true
argument-hint: "[the question or problem to brainstorm]"
---

# Help Me Brainstorm

This skill widens the user's options on an open question, then narrows them to one or two chosen ideas. It has three phases: frame, diverge, and converge. During the diverge phase, Claude generates ideas freely and does not criticise them. Pushback waits until the converge phase.

## Rules that always apply

1. Ask exactly one question per AskUserQuestion call. The questions array must contain one item, never more.
2. Every multiple choice question has two to four options. Put your recommended option first and add "(Recommended)" to its label. Give every option a short description of its main trade off.
3. The user can always add their own thoughts. AskUserQuestion always includes an "Other" box, so remind the user once, at the start, that they can type their own answer there. Never force the user to contribute ideas.
4. Do not write plans, code, or files. Everything stays in the conversation.
5. Use plain, simple language in full sentences.

## Step 1: Get the question

If the user gave a question with the command ($ARGUMENTS is not empty), restate it in one sentence.

If not, ask in plain text: "What question or problem do you want to brainstorm?"

## Step 2: Ground the question (only inside a repo)

If the current directory is a repo, look for code, notebooks, data files, and docs related to the question. For broad searches, use the Explore subagent. Note what already exists and what has already been tried.

Send the user a short summary of two to four sentences with file paths. If nothing relevant is found, say so in one sentence and continue.

If there is no repo, skip this step and work from the conversation alone.

## Step 3: Frame the problem

Ask three to four short framing questions, one at a time. Use AskUserQuestion with suggested options where sensible, and plain text where there is no meaningful set of options. Cover:

1. **Goal.** What outcome does the user want?
2. **Success metric.** How would they know an idea worked? Suggest likely metrics for the domain.
3. **Constraints.** Time, budget, data, team, or tooling limits.
4. **Already tried.** What has been tried before, and what happened? Use what you found in Step 2 to suggest options.

Skip any question the user already answered. Keep framing short. Do not grill the user here.

## Step 4: Diverge

Generate 8 to 12 ideas. Do not criticise any idea in this step.

1. Choose three or four lenses that fit the problem. For ad creative performance, these could be creative content, targeting, measurement, and modelling.
2. Group the ideas under their lens.
3. Tag each idea as **Quick win** or **Larger bet**.
4. Give each idea one line on why it might work.
5. Include at least one idea that challenges the framing itself, such as questioning the success metric or the scope.
6. Avoid ideas the user said were already tried, unless you explain what would be different this time.

Show the list as text, numbered so the user can refer to ideas by number. End the list with this invitation: "Add, change, or combine any ideas before we narrow down, or say 'next' to move on."

If the user asks for more ideas in a lens, add them and show the invitation again.

## Step 5: Converge

1. Score every idea low, medium, or high on three criteria:
   1. **Impact** on the success metric from Step 3.
   2. **Effort** to try it.
   3. **Confidence** that it will work. Include whether the result can be measured cleanly, for example in an A/B test.
2. Show the scores in a compact table with one short reason for each score that is not obvious.
3. Use AskUserQuestion to ask the user which idea they want to pursue. Offer the three or four strongest ideas as options, with your top pick recommended. The user can type any other idea number in the "Other" box. Then ask whether they want to keep a second idea.
4. Push back on the chosen ideas. Check for:
   1. Effort or confidence the user may be underestimating.
   2. Conflicts with the constraints from Step 3, or with what was already tried.
   3. A risk to measurement, such as no clear baseline, a weak metric, or not enough traffic for an A/B test.
   4. Weak reasoning. If the choice seems unexplained, ask why.

   Raise the single most important concern as a question. Let the user keep or change their choice.

## Step 6: Summary

Write a short summary in the conversation. Do not save it to a file. Use these sections:

1. **Question and framing.** The question, goal, success metric, constraints, and what was already tried.
2. **Chosen ideas.** The one or two chosen ideas, with the user's reasons in their own words where possible.
3. **Ideas worth keeping.** Two to four runner up ideas, each with one line on why they are worth revisiting.

Then stop. Do not offer a plan or next steps unless the user asks.
