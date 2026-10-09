"""Support layer — P4 (explanations and optional suggestions from the
screening answers the person already gave).

There is deliberately **no second questionnaire** in this package: the report is
assembled from the episode result, its explanation and the recorded answers.

Modules:
    questionnaire  the instrument, derived from the verified feature contract
    schemas        Pydantic contracts (support-report/2.0)
    domains        which items may justify which area, and what is unmeasured
    strategies     curated content with provenance and review status
    engine         deterministic rules: assessments -> suggestions
    report         assembly of the validated ScreeningReport
"""
