"""Support layer — P4 (explainable, support-oriented screening).

Modules:
    schemas      Pydantic data contracts (validated at every boundary)
    domains      candidate support domains and their evidence linkage
    questions    the optional follow-up questionnaire (neutral wording)
    strategies   curated support strategies with provenance
    engine       deterministic recommendation rules
    report       assembly of the validated ScreeningReport
"""
