## Software Fundamentals

These principles come from Matt Pocock's talk on software fundamentals with AI. The goal is a codebase that stays easy to change, because AI works well in a good codebase and poorly in a bad one.

### Reach a shared understanding when I ask for it
* Only start this interview when I explicitly ask for it, for example by saying "grill me", "help me plan", "interview me", or "ask me questions about this". Otherwise, proceed with the task as normal.
* When triggered, interview me about the plan. Ask one question at a time and work through each branch of decisions, resolving dependencies between them in order.
* Keep asking until we share the same picture of what we are building. Do not rush to produce a plan or start implementing.
* When we agree, summarize the result as a short requirements document or a list of issues.

### Use a shared vocabulary
* If the project has a glossary file (for example `UBIQUITOUS_LANGUAGE.md` or `GLOSSARY.md`), read it first and use its terms exactly, in code, plans, and conversation.
* If a new domain term comes up, propose adding it to the glossary rather than inventing a synonym.
* Keep explanations short and use the agreed terms instead of long descriptions.

### Take small steps with fast feedback
* The speed of feedback sets the safe speed of work. Do not write large amounts of code before checking it.
* Prefer test driven development: write a failing test, make it pass with the smallest change, then refactor.
* Run type checks, linters, and tests after each small step, not only at the end.
* For front end work, check the result in a browser when a browser tool is available.
* For critical logic, show me the implementation in full. In ML work this includes data splits, label construction, leakage risks, feature transformations shared between training and serving, and evaluation metrics.

### Prefer deep modules
* A deep module hides a lot of functionality behind a small, simple interface. A shallow module exposes many small functions and offers little in return. Prefer deep modules.
* Do not split logic into many tiny files or helpers that callers must combine in the right order.
* Test behaviour at the module interface rather than testing every internal helper.
* When code that belongs together is scattered, suggest grouping it into one deep module.

### Structure ML systems as feature, training, and inference pipelines
These rules apply only when building an ML system that goes beyond a one off analysis or notebook. They follow the FTI pipeline approach described by Hopsworks.
* Split the system into three independent pipelines, each with a clear input and output. Treat each pipeline as a deep module.
  * **Feature pipeline:** reads raw data and writes features and labels to a feature store.
  * **Training pipeline:** reads point in time correct training data from the feature store and writes a trained model to a model registry.
  * **Inference pipeline:** reads new features and a trained model and writes predictions to a consumer, such as a table, dashboard, or API.
* Pipelines communicate only through the feature store and the model registry. Do not let one pipeline call into another pipeline's internals.
* Define feature transformations once and reuse the same code in training and inference, to avoid training and serving skew.
* Store features in a reusable form where possible, and apply model specific encoding and scaling in the training and inference pipelines.
* Version models together with the feature versions they depend on, so they can be rolled back together.
* Test in layers: unit test feature logic, validate input data, evaluate model performance and bias, and use A/B tests before replacing a model in production.
* For online inference, keep request time feature computation small and fetch precomputed features by entity ID.
* When planning, state which of the three pipelines a change touches.

### Ground ML design recommendations in standard practice
* When recommending or offering options for ML experimentation design or ML system design, start from the industry standard pattern, then adapt it to this project's requirements.
* State both parts explicitly: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up.
* If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.

### Invest in the design every day
* In plans and requirements documents, state which modules and interfaces will change and how.
* Consider the design of the whole system with each change, not only the change itself. Point out when a change would make the codebase harder to change later.
* Code is not cheap. Leave the codebase easier to change than you found it.
