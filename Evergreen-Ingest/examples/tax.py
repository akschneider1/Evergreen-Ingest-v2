import langextract as lx
from examples import DOMAINS, DomainConfig

PROMPT = """Extract discrete policy parameters from this government tax document.
For each parameter found, identify its type, the exact source text as it appears
in the document, and its structured values. Only extract parameters that are
explicitly stated with specific values, amounts, dates, or rates.
Skip narrative, background context, and general descriptions."""

EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "Colorado imposes a flat income tax rate of 4.4% on Colorado taxable income "
            "for tax year 2026. "
            "Colorado residents with gross income exceeding $14,600 for single filers "
            "must file a state return by April 15. "
            "The penalty for failure to file is 5% of the unpaid tax per month, "
            "up to a maximum of 25%."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="tax_rate",
                extraction_text=(
                    "flat income tax rate of 4.4% on Colorado taxable income for tax year 2026"
                ),
                attributes={
                    "parameter": "state income tax rate",
                    "value": "4.4%",
                    "applies_to": "all filing statuses",
                },
            ),
            lx.data.Extraction(
                extraction_class="filing_threshold",
                extraction_text="gross income exceeding $14,600 for single filers",
                attributes={
                    "parameter": "filing threshold",
                    "value": "$14,600",
                    "applies_to": "single filers",
                },
            ),
            lx.data.Extraction(
                extraction_class="penalty_rule",
                extraction_text=(
                    "penalty for failure to file is 5% of the unpaid tax per month, "
                    "up to a maximum of 25%"
                ),
                attributes={
                    "parameter": "failure-to-file penalty",
                    "rate": "5% per month",
                    "maximum": "25%",
                },
            ),
        ],
    ),
]

DOMAINS["tax"] = DomainConfig(
    name="tax",
    prompt=PROMPT,
    examples=EXAMPLES,
)
