"""
Submit Anthropic's one-time "first-time use" form for Bedrock via the API,
then check whether the agreement has gone through.

Run (venv active, same terminal as your exports):
  python submit_use_case.py submit     # sends the form (only needs doing once per account)
  python submit_use_case.py check      # checks agreement status; rerun until AVAILABLE

Edit FORM below so it's accurate before you submit.
"""

import json
import os
import sys

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
CHECK_MODEL = "anthropic.claude-sonnet-5-5"  # any Anthropic model ID from list-foundation-models

FORM = {
    "companyName": "Ntusekho Mngadi (individual developer)",
    "companyWebsite": "https://www.linkedin.com/in/ntusekho-mngadi-7932b7125",
    "intendedUsers": "0",            # 0 = internal, 1 = external, 2 = both
    "industryOption": "Technology",
    "otherIndustryOption": "",
    "useCases": (
        "Personal learning and prototyping: building AI agents with the Bedrock "
        "Converse API and tool use, including an AWS cost-analysis assistant "
        "that reads Cost Explorer data. Non-production sandbox account."
    ),
}

# Control-plane client ("bedrock", not "bedrock-runtime")
client = boto3.client("bedrock", region_name=REGION)


def submit():
    try:
        client.put_use_case_for_model_access(formData=json.dumps(FORM))
        print("Submitted. Now run: python submit_use_case.py check")
    except ClientError as e:
        print(f"{e.response['Error']['Code']}: {e.response['Error']['Message']}")
        print("If it complains about a field value, paste me the error and we'll adjust FORM.")


def check():
    resp = client.get_foundation_model_availability(modelId=CHECK_MODEL)
    resp.pop("ResponseMetadata", None)
    print(json.dumps(resp, indent=2))
    status = resp["agreementAvailability"]["status"]
    print(f"\nagreement status -> {status}")
    if status == "AVAILABLE":
        print("You're in. export BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-5 (or a 5.5 profile) and go.")
    elif status == "PENDING":
        print("Submitted, waiting on approval. Watch your email and check again later.")
    else:
        print(
            "No agreement yet. If you already submitted the form, try one Claude converse call to trigger\n"
            "the subscription; if it's still NOT_AVAILABLE after a day, open an AWS support case."
        )


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"submit": submit, "check": check}.get(action, check)()
