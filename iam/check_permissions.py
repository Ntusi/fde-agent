"""
Prove the agent's IAM policy works in BOTH directions
=====================================================

A least-privilege policy is only proven when you have seen it ALLOW what the agent
needs and DENY what it doesn't. This script tries both and reports.

  AWS_PROFILE=fde-agent   python iam/check_permissions.py     (the agent's role: all checks should PASS)
  AWS_PROFILE=fde-sandbox python iam/check_permissions.py     (admin: the "should be denied" checks FAIL,
                                                               which shows you what admin would have allowed)

Cost: about $0.01 (one Cost Explorer request) plus a few tokens. Every "should be
denied" call is read-only, so even under admin nothing is changed.
"""

import datetime as dt
import os
import sys

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
ALLOWED_MODEL = "us.amazon.nova-2-lite-v1:0"
OTHER_MODEL = "us.amazon.nova-pro-v1:0"  # deliberately NOT in the policy
DENIED_CODES = {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation", "UnauthorizedException"}

session = boto3.Session()
bedrock_rt = session.client("bedrock-runtime", region_name=REGION)
bedrock_cp = session.client("bedrock", region_name=REGION)
ce = session.client("ce", region_name="us-east-1")
s3 = session.client("s3", region_name=REGION)
ec2 = session.client("ec2", region_name=REGION)
iam = session.client("iam")
sts = session.client("sts")

HI = [{"role": "user", "content": [{"text": "Reply with the single word OK."}]}]
SMALL = {"maxTokens": 10, "temperature": 0}


def last_month():
    first_of_this = dt.date.today().replace(day=1)
    first_of_last = (first_of_this - dt.timedelta(days=1)).replace(day=1)
    return {"Start": first_of_last.isoformat(), "End": first_of_this.isoformat()}


def stream_once():
    resp = bedrock_rt.converse_stream(modelId=ALLOWED_MODEL, messages=HI, inferenceConfig=SMALL)
    for _ in resp["stream"]:  # read the whole stream, like Strands does
        pass


# (description, expectation, call)
CHECKS = [
    ("Converse on Nova 2 Lite (hand-built agent)", "allow",
     lambda: bedrock_rt.converse(modelId=ALLOWED_MODEL, messages=HI, inferenceConfig=SMALL)),
    ("ConverseStream on Nova 2 Lite (Strands agent)", "allow", stream_once),
    ("Cost Explorer GetCostAndUsage (the tools)", "allow",
     lambda: ce.get_cost_and_usage(TimePeriod=last_month(), Granularity="MONTHLY", Metrics=["UnblendedCost"])),
    ("Converse on Nova Pro (a model not in the policy)", "deny",
     lambda: bedrock_rt.converse(modelId=OTHER_MODEL, messages=HI, inferenceConfig=SMALL)),
    ("Bedrock ListFoundationModels (control plane)", "deny", lambda: bedrock_cp.list_foundation_models()),
    ("Cost Explorer GetCostForecast (another CE action)", "deny",
     lambda: ce.get_cost_forecast(
         TimePeriod={"Start": (dt.date.today() + dt.timedelta(days=1)).isoformat(),
                     "End": (dt.date.today() + dt.timedelta(days=8)).isoformat()},
         Metric="UNBLENDED_COST", Granularity="MONTHLY")),
    ("S3 ListBuckets", "deny", lambda: s3.list_buckets()),
    ("EC2 DescribeInstances", "deny", lambda: ec2.describe_instances(MaxResults=5)),
    ("IAM ListUsers", "deny", lambda: iam.list_users(MaxItems=1)),
]


def run_check(call):
    try:
        call()
        return "allowed", ""
    except ClientError as e:
        code = e.response["Error"]["Code"]
        return ("denied" if code in DENIED_CODES else "error"), code


def main():
    who = sts.get_caller_identity()["Arn"]  # always works, whatever the policy
    role = who.split("/")[1] if "/" in who else who
    print(f"Running as: {role}\n")
    results = []
    for desc, expect, call in CHECKS:
        outcome, code = run_check(call)
        ok = (expect == "allow" and outcome == "allowed") or (expect == "deny" and outcome == "denied")
        results.append(ok)
        mark = "PASS" if ok else "FAIL"
        detail = f"{outcome}" + (f" ({code})" if code else "")
        print(f"  {mark}  should {expect:<5}  {desc:<52} -> {detail}")
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks as expected.")
    if passed == len(results):
        print("Least privilege proven: the agent can do its job and nothing else.")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
