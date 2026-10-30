---
name: help-me-understand
description: Helps the user understand any topic, file, folder, link, or plan. First writes a short explanation of the foundations to build intuition, then quizzes the user with one multiple choice question at a time using AskUserQuestion, marking each answer and explaining it. Add "quiz only" to skip the explanation. Use only when the user types /help-me-understand.
disable-model-invocation: true
argument-hint: "[topic, file, folder, link, or 'the plan above'] [quiz only]"
---

# Help Me Understand

This skill builds and checks the user's understanding of a subject. It starts with a short explanation of the foundations and the big picture, to build intuition. Then it runs an interview style quiz, asking one multiple choice question at a time, marking each answer with an emoji, and explaining the correct answer whenever the user is wrong or unsure. The quiz adapts as the user answers. The user can skip the explanation and go straight to the quiz.

## Rules that always apply

1. Ask exactly one question per AskUserQuestion call. The questions array must contain one item, never more.
2. Every question has exactly four options: three answer options and a final option labelled "I don't know".
3. Exactly one answer option is correct. The other two must be plausible mistakes that someone with a partial understanding might make, not obviously silly choices.
4. Never mark any option as "(Recommended)", and never hint at the correct answer in labels or descriptions. Option descriptions should add context to the choice, not judge it.
5. Change the position of the correct answer from question to question. It must not follow a pattern.
6. The "Other" box is always available. Remind the user once, just before the first quiz question, that they can type their own answer there.
7. Every fact used in the explanation, a question, or a marking must be accurate. If you are not sure of a fact, check the source or search before using it.
8. Use plain, simple language in full sentences.

## Step 1: Identify the subject

Read what the user typed after the command ($ARGUMENTS).

If the arguments contain "quiz only", or otherwise clearly ask to skip the explanation, use **quiz only mode**: skip Step 3. Remove that phrase before working out the subject.

1. **A file or folder.** Read it. For a large folder or repo, use the Explore subagent to find the main parts.
2. **A link.** Fetch it with WebFetch.
3. **Something earlier in the conversation**, such as a plan or a decision summary. Use that content.
4. **A topic.** Use your own knowledge for stable topics. For recent, niche, or fast changing topics, run a web search first to confirm the facts.
5. **Nothing.** Ask in plain text: "What do you want to be quizzed on?"

Restate the subject in one sentence before the explanation, or before the first question in quiz only mode.

## Step 2: Map the key ideas

Privately list the key ideas in the subject, ordered from basic to advanced. Note which ideas depend on others. Mark each idea as a **foundation** (the basics and the big picture of how they fit together) or **advanced**. This list decides the explanation, the questions, and when the quiz ends. Do not show it to the user.

## Step 3: Build intuition first

Skip this step in quiz only mode.

Write an explanation in free prose of about 500 words or fewer. Cover only the foundation ideas and how they fit together. Leave the advanced ideas for the quiz. Shape the prose to suit the subject, but always:

1. Say why the subject matters.
2. Include at least one concrete example or analogy.
3. Name one common confusion and explain why it is wrong.

For code, a file, or a folder, explain what the code is for and how its parts work together, not a list of what each file contains.

End the turn with one plain text line, such as "Ask me anything about this, or say **go** to start the quiz." Do not call AskUserQuestion in the same turn. If the user asks follow up questions, answer them, then offer "go" again. Start the quiz when the user says go or something like it.

## Step 4: Run the quiz

For each question, write one or two sentences of context in the question text when it helps. The wrong options may reuse the common confusion named in the explanation.

### Marking each answer

After each answer, reply with one of the following, then move on to the next question.

1. **Correct.** Start with ✅, then one sentence on why it is right.
2. **Wrong.** Start with ❌, say the answer was wrong, state the correct answer, and explain it in two to four sentences. Also explain briefly why the chosen option was tempting but wrong.
3. **Partly correct typed answer.** Start with 🟡, say what the user got right and what they missed, then explain the full correct answer.
4. **Correct typed answer.** Start with ✅ and confirm it in one sentence. Add anything important the user left out.
5. **"I don't know".** Start with 💡, then explain the correct answer in two to four sentences, as you would to someone meeting the idea for the first time.

### Adapting difficulty

1. After the explanation, start with one or two quick questions on the foundation ideas to check that the explanation landed, then move to the advanced ideas. Foundation ideas covered by the explanation and confirmed by these questions count as tested. In quiz only mode, start with basic questions and test the foundation ideas one by one.
2. After two correct answers in a row, move to a harder idea or a harder question on the same idea.
3. After a wrong answer or "I don't know", stay at the current level. Within the next few questions, test the same idea again from a different angle, such as a new example or the reverse question.
4. Do not ask about an idea until the ideas it depends on have been tested.

### Keeping the user oriented

About every five questions, send a short progress note with the score so far and roughly how many key ideas remain. The user can type "stop" at any time to go straight to Step 5.

## Step 5: End with a summary

The quiz ends when every key idea has been tested, including the follow up questions on missed ideas, or when the user says stop. Write a short summary in the conversation with these sections:

1. **Score.** Correct answers out of total questions.
2. **Ideas you know well.** The key ideas the user answered correctly.
3. **Ideas worth reviewing.** The key ideas the user got wrong, partly right, or did not know, each with one sentence on the main point to remember.

Then stop.
