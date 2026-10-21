---
name: help-me-understand
description: Quizzes the user on any topic, file, folder, link, or plan with one multiple choice question at a time using AskUserQuestion, then marks each answer and explains it. Use only when the user types /help-me-understand.
disable-model-invocation: true
argument-hint: "[topic, file, folder, link, or 'the plan above']"
---

# Help Me Understand

This skill runs an interview style quiz to check and build the user's understanding of a subject. It asks one multiple choice question at a time, marks each answer with an emoji, and explains the correct answer whenever the user is wrong or unsure. It starts with the basics and adapts as the user answers.

## Rules that always apply

1. Ask exactly one question per AskUserQuestion call. The questions array must contain one item, never more.
2. Every question has exactly four options: three answer options and a final option labelled "I don't know".
3. Exactly one answer option is correct. The other two must be plausible mistakes that someone with a partial understanding might make, not obviously silly choices.
4. Never mark any option as "(Recommended)", and never hint at the correct answer in labels or descriptions. Option descriptions should add context to the choice, not judge it.
5. Change the position of the correct answer from question to question. It must not follow a pattern.
6. The "Other" box is always available. Remind the user once, at the start, that they can type their own answer there.
7. Every fact used in a question or explanation must be accurate. If you are not sure of a fact, check the source or search before asking about it.
8. Use plain, simple language in full sentences.

## Step 1: Identify the subject

Read what the user typed after the command ($ARGUMENTS).

1. **A file or folder.** Read it. For a large folder or repo, use the Explore subagent to find the main parts.
2. **A link.** Fetch it with WebFetch.
3. **Something earlier in the conversation**, such as a plan or a decision summary. Use that content.
4. **A topic.** Use your own knowledge for stable topics. For recent, niche, or fast changing topics, run a web search first to confirm the facts.
5. **Nothing.** Ask in plain text: "What do you want to be quizzed on?"

Restate the subject in one sentence before the first question.

## Step 2: Map the key ideas

Privately list the key ideas in the subject, ordered from basic to advanced. Note which ideas depend on others. This list decides the questions and when the quiz ends. Do not show it to the user.

## Step 3: Run the quiz

Start with the most basic key idea. For each question, write one or two sentences of context in the question text when it helps.

### Marking each answer

After each answer, reply with one of the following, then move on to the next question.

1. **Correct.** Start with ✅, then one sentence on why it is right.
2. **Wrong.** Start with ❌, say the answer was wrong, state the correct answer, and explain it in two to four sentences. Also explain briefly why the chosen option was tempting but wrong.
3. **Partly correct typed answer.** Start with 🟡, say what the user got right and what they missed, then explain the full correct answer.
4. **Correct typed answer.** Start with ✅ and confirm it in one sentence. Add anything important the user left out.
5. **"I don't know".** Start with 💡, then explain the correct answer in two to four sentences, as you would to someone meeting the idea for the first time.

### Adapting difficulty

1. Start with basic questions.
2. After two correct answers in a row, move to a harder idea or a harder question on the same idea.
3. After a wrong answer or "I don't know", stay at the current level. Within the next few questions, test the same idea again from a different angle, such as a new example or the reverse question.
4. Do not ask about an idea until the ideas it depends on have been tested.

### Keeping the user oriented

About every five questions, send a short progress note with the score so far and roughly how many key ideas remain. The user can type "stop" at any time to go straight to Step 4.

## Step 4: End with a summary

The quiz ends when every key idea has been tested, including the follow up questions on missed ideas, or when the user says stop. Write a short summary in the conversation with these sections:

1. **Score.** Correct answers out of total questions.
2. **Ideas you know well.** The key ideas the user answered correctly.
3. **Ideas worth reviewing.** The key ideas the user got wrong, partly right, or did not know, each with one sentence on the main point to remember.

Then stop.
