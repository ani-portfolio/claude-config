"""Project checks for the V1 bar. ml-problem-framing creates this file once, the first time a V1 criterion of kind
check is used, and never changes it, so it is yours to edit.

A criterion in framing.success.v1_criteria such as

    - {id: media_nonneg, kind: check, function: media_coefs_nonneg, on: cv}

calls the function of that name here when model training (on: cv) or offline evaluation (on: test) judges the
V1 bar. Optional params in the criterion are passed as keyword arguments.

Each function takes a context and returns (passed, detail): True or False, and one plain sentence. Return None
instead of True or False when the check cannot be judged; the bar is then reported incomplete, not failed.

The context (model_core.V1Context) gives:
    ctx.spec, ctx.root, ctx.on          the spec, the project root, and "cv" or "test"
    ctx.model                           the champion's definition: name, type, params, inputs
    ctx.data, ctx.inputs, ctx.target    the training feature table, the input columns, and the target
    ctx.pipeline                        the champion fitted on every training row (made when first used)
    ctx.fold_pipelines                  the champion fitted on each CV fold's training rows (made when first used)
    ctx.coefficients(pipe, raw_units)   a linear model's coefficients by input name, or None for other models
    ctx.test                            the test rows, only when on is "test"

Checks must not use the test rows unless the criterion says on: test.
"""


def example_check(ctx):
    """A template: passes when the champion is a linear model. Replace it with the project's own checks."""
    coefs = ctx.coefficients()
    if coefs is None:
        return None, "The champion is not a linear model, so this check does not apply."
    return True, f"The champion has {len(coefs)} coefficients."
