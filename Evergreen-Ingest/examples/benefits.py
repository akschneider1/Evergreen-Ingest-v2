import langextract as lx
from examples import DOMAINS, DomainConfig

PROMPT = """Extract discrete policy parameters from this government benefits document.
For each parameter found, identify its type, the exact source text as it appears
in the document, and its structured values. Only extract parameters that are
explicitly stated with specific values, amounts, dates, thresholds, or requirements.
Skip narrative explanation and general eligibility descriptions without specific values."""

EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "The weekly benefit amount (WBA) is 60% of the claimant's average weekly "
            "wage during the two highest-earning quarters. Maximum WBA is $781 per week "
            "effective January 1, 2026. "
            "Claimants must conduct at least 5 employer contacts per week. "
            "Claims must be filed within 3 weeks of the first week of unemployment."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="benefit_amount",
                extraction_text="Maximum WBA is $781 per week effective January 1, 2026",
                attributes={
                    "parameter": "maximum weekly benefit amount",
                    "value": "$781",
                    "effective_date": "January 1, 2026",
                },
            ),
            lx.data.Extraction(
                extraction_class="work_search_requirement",
                extraction_text="at least 5 employer contacts per week",
                attributes={
                    "parameter": "weekly work search contacts",
                    "value": "5 contacts",
                },
            ),
            lx.data.Extraction(
                extraction_class="enrollment_period",
                extraction_text=(
                    "Claims must be filed within 3 weeks of the first week of unemployment"
                ),
                attributes={
                    "parameter": "initial claim filing window",
                    "value": "3 weeks",
                },
            ),
        ],
    ),
]

DOMAINS["benefits"] = DomainConfig(
    name="benefits",
    prompt=PROMPT,
    examples=EXAMPLES,
)
