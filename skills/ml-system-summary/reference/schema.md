# system_summary.json Format

The diagram builder reads this file. See `examples/system_summary.example.json` for a complete example.

```json
{
  "system_name": "Short descriptive name of the system",
  "repo": "Repository name or path",
  "generated_on": "YYYY-MM-DD",
  "discrepancy_count": 0,
  "framing": {
    "business_goal": ["One or two short bullets"],
    "assumptions": ["Short bullets"],
    "problem_framing": [
      "Unit of prediction: ...",
      "Prediction horizon: ...",
      "Label: ...",
      "Task type: ...",
      "Offline metric: ...",
      "Serving: ...",
      "Downstream decision: ...",
      "Business KPI: ...",
      "Guardrails: ...",
      "Online test: ..."
    ]
  },
  "components": {
    "<component key>": {
      "status": "confirmed | inferred | documented | missing",
      "bullets": ["Short bullet", "..."],
      "evidence": ["path/to/code_file.py (function_name)"]
    }
  },
  "feedback_loops": {
    "retraining_trigger": {"present": true, "label": "Short label for the arrow"},
    "behaviour_feedback": {"present": false, "label": ""}
  }
}
```

## Component keys

Every key should be present. Use `"status": "missing"` with no bullets when a component was not found.

| Key | Diagram box | Color |
|---|---|---|
| `raw_data` | Raw Data | Blue (data) |
| `feature_pipeline` | Feature Pipeline | Green (pipeline) |
| `engineered_features` | Engineered Features | Blue (data) |
| `feature_store` | Feature Store | Pink (infrastructure) |
| `training_pipeline` | Training Pipeline | Green (pipeline) |
| `trained_model` | Trained Model | Blue (data) |
| `model_registry` | Model Registry | Pink (infrastructure) |
| `deployment_experimentation` | Deployment and Experimentation | Pink (infrastructure) |
| `inference_pipeline` | Inference Pipeline | Green (pipeline) |
| `predictions_decision` | Predictions and Decision | Yellow (output) |
| `monitoring` | Monitoring and Observability | Pink (infrastructure) |

## Rules

1. Keep at most six bullets per component and about ten words per bullet. The builder shortens longer lists and notes how many items were left out.
2. Framing bullets are not shortened, but keep them brief so the left panel stays readable.
3. Bullets describe what the code does. Evidence lists code files only, never documentation files.
4. `inferred` adds "(inferred)" to the box title. `documented` draws a grey dashed box titled "(docs only, not in code)" and keeps its bullets, so say what the documentation claims and that no code was found. `missing` draws a grey dashed box with "Not found in codebase". Arrows touching a `documented` or `missing` box become grey and dashed.
5. `discrepancy_count` is the number of High impact rows in the Documentation and Code Discrepancies section. When it is above zero, a red note above the diagram says the diagram follows the code.
6. Only set a feedback loop to `"present": true` when the code or config supports it.
7. Data quality, orchestration, discrepancies, and gaps appear only in the markdown, not in the diagram.
