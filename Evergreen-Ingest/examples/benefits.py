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
            "To qualify for unemployment benefits, total base period wages must be "
            "at least $2,500. Wages in the highest-earning quarter must be at least "
            "$1,800. Total base period wages must be at least 1.25 times the "
            "high-quarter wages."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="eligibility_rule",
                extraction_text="total base period wages must be at least $2,500",
                attributes={
                    "parameter": "minimum base period wages",
                    "value": "$2,500",
                    "applies_to": "all claimants",
                },
            ),
            lx.data.Extraction(
                extraction_class="eligibility_rule",
                extraction_text=(
                    "Wages in the highest-earning quarter must be at least $1,800"
                ),
                attributes={
                    "parameter": "minimum high-quarter wages",
                    "value": "$1,800",
                    "applies_to": "all claimants",
                },
            ),
            lx.data.Extraction(
                extraction_class="eligibility_rule",
                extraction_text=(
                    "Total base period wages must be at least 1.25 times the high-quarter wages"
                ),
                attributes={
                    "parameter": "wage spread requirement",
                    "value": "1.25x high-quarter wages",
                    "applies_to": "all claimants",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The weekly benefit amount (WBA) is 60% of the claimant's average weekly "
            "wage during the two highest-earning quarters. Minimum WBA is $25 per week. "
            "Maximum WBA is $781 per week effective January 1, 2026."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="benefit_amount",
                extraction_text=(
                    "60% of the claimant's average weekly wage during the two "
                    "highest-earning quarters"
                ),
                attributes={
                    "parameter": "weekly benefit amount formula",
                    "value": "60% of average weekly wage",
                    "basis": "two highest-earning quarters",
                },
            ),
            lx.data.Extraction(
                extraction_class="benefit_amount",
                extraction_text="Minimum WBA is $25 per week",
                attributes={
                    "parameter": "minimum weekly benefit amount",
                    "value": "$25",
                },
            ),
            lx.data.Extraction(
                extraction_class="benefit_amount",
                extraction_text="Maximum WBA is $781 per week effective January 1, 2026",
                attributes={
                    "parameter": "maximum weekly benefit amount",
                    "value": "$781",
                    "effective_date": "January 1, 2026",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "Claimants must conduct at least 5 employer contacts per week, at least "
            "3 of which must be direct applications, recorded in the MyUI+ portal "
            "by midnight Sunday. Refusing a suitable offer of work without good cause "
            "results in disqualification for 8 weeks."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="work_search_requirement",
                extraction_text=(
                    "at least 5 employer contacts per week, at least 3 of which must "
                    "be direct applications, recorded in the MyUI+ portal by midnight Sunday"
                ),
                attributes={
                    "parameter": "weekly work search contacts",
                    "total_contacts": "5",
                    "direct_applications": "3",
                    "deadline": "midnight Sunday",
                    "portal": "MyUI+",
                },
            ),
            lx.data.Extraction(
                extraction_class="disqualification_rule",
                extraction_text=(
                    "Refusing a suitable offer of work without good cause results in "
                    "disqualification for 8 weeks"
                ),
                attributes={
                    "parameter": "suitable work refusal penalty",
                    "value": "8 weeks disqualification",
                    "condition": "refusal without good cause",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "Claims must be filed within 3 weeks of the first week of unemployment. "
            "There is no minimum age requirement for unemployment benefits. "
            "Expedited claims must be resolved within 5 business days of the request. "
            "Expedited service is available when household income is below 50% of the "
            "federal poverty level."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="enrollment_period",
                extraction_text=(
                    "Claims must be filed within 3 weeks of the first week of unemployment"
                ),
                attributes={
                    "parameter": "initial claim filing window",
                    "value": "3 weeks",
                    "from": "first week of unemployment",
                },
            ),
            lx.data.Extraction(
                extraction_class="eligibility_rule",
                extraction_text=(
                    "There is no minimum age requirement for unemployment benefits"
                ),
                attributes={
                    "parameter": "minimum age requirement",
                    "value": "none",
                    "applies_to": "all workers with qualifying employment",
                },
            ),
            lx.data.Extraction(
                extraction_class="enrollment_period",
                extraction_text=(
                    "Expedited claims must be resolved within 5 business days of the request"
                ),
                attributes={
                    "parameter": "expedited claim resolution window",
                    "value": "5 business days",
                    "condition": "expedited request",
                },
            ),
            lx.data.Extraction(
                extraction_class="eligibility_rule",
                extraction_text=(
                    "Expedited service is available when household income is below "
                    "50% of the federal poverty level"
                ),
                attributes={
                    "parameter": "expedited service income threshold",
                    "value": "below 50% federal poverty level",
                    "applies_to": "expedited service claimants",
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
