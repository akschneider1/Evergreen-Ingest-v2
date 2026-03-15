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
            "Colorado residents with gross income exceeding $14,600 for single filers "
            "must file a state return by April 15."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="filing_threshold",
                extraction_text="gross income exceeding $14,600 for single filers",
                attributes={
                    "parameter": "filing threshold",
                    "value": "$14,600",
                    "applies_to": "single filers",
                    "effective_year": "2026",
                },
            ),
            lx.data.Extraction(
                extraction_class="filing_deadline",
                extraction_text="must file a state return by April 15",
                attributes={
                    "parameter": "filing deadline",
                    "value": "April 15",
                    "applies_to": "all filers",
                    "extension_available": "yes — 6 months to October 15",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The penalty for failure to file is 5% of the unpaid tax per month, "
            "up to a maximum of 25%. The minimum penalty for returns more than "
            "60 days late is $25 or 10% of the tax due, whichever is greater."
        ),
        extractions=[
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
                    "penalty_type": "failure to file",
                },
            ),
            lx.data.Extraction(
                extraction_class="penalty_rule",
                extraction_text=(
                    "minimum penalty for returns more than 60 days late is $25 or "
                    "10% of the tax due, whichever is greater"
                ),
                attributes={
                    "parameter": "minimum late-file penalty",
                    "value": "$25 or 10% of tax due",
                    "condition": "more than 60 days late",
                    "penalty_type": "failure to file — minimum",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "Colorado's EITC is 38% of the federal EITC for tax year 2026. "
            "This credit is fully refundable. "
            "Colorado offers a refundable Child Tax Credit equal to 25% of the "
            "federal child tax credit amount, available to taxpayers with income "
            "below $75,000 (single) or $85,000 (joint)."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="credit",
                extraction_text="Colorado's EITC is 38% of the federal EITC for tax year 2026",
                attributes={
                    "parameter": "Earned Income Tax Credit",
                    "value": "38% of federal EITC",
                    "refundable": "yes",
                    "effective_year": "2026",
                },
            ),
            lx.data.Extraction(
                extraction_class="credit",
                extraction_text=(
                    "Child Tax Credit equal to 25% of the federal child tax credit amount, "
                    "available to taxpayers with income below $75,000 (single) or $85,000 (joint)"
                ),
                attributes={
                    "parameter": "Child Tax Credit",
                    "value": "25% of federal Child Tax Credit",
                    "income_limit_single": "$75,000",
                    "income_limit_joint": "$85,000",
                    "refundable": "yes",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "Colorado imposes a flat income tax rate of 4.4% on Colorado taxable income "
            "for tax year 2026. "
            "Employers must apply the 22% federal supplemental withholding rate to "
            "supplemental wages unless the employee has submitted a W-4 electing a "
            "different rate."
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
                    "rate_type": "flat",
                    "applies_to": "all filing statuses",
                    "effective_year": "2026",
                },
            ),
            lx.data.Extraction(
                extraction_class="tax_rate",
                extraction_text=(
                    "22% federal supplemental withholding rate to supplemental wages"
                ),
                attributes={
                    "parameter": "federal supplemental withholding rate",
                    "value": "22%",
                    "applies_to": "supplemental wages (bonuses, commissions, overtime)",
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
