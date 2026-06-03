from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbond_on.app.usecases.ai_factor_factory import (
    build_review_cfg_for_request,
    build_dify_inputs,
    generate_from_dify,
    stage_candidate_file,
    validate_candidate_file,
    write_candidate_package,
    review_candidate,
)


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI factor factory research-only tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prompt = sub.add_parser("render-dify-inputs", help="Print Dify workflow input payload")
    p_prompt.add_argument("--topic", required=True)
    p_prompt.add_argument("--constraints", default="")
    p_prompt.add_argument("--batch-id", default="")

    p_gen = sub.add_parser("generate-dify", help="Call Dify workflow and stage returned candidates")
    p_gen.add_argument("--topic", required=True)
    p_gen.add_argument("--constraints", default="")
    p_gen.add_argument("--batch-id", default="")

    p_validate = sub.add_parser("validate", help="Validate a candidate JSON file")
    p_validate.add_argument("--candidate", required=True)

    p_stage = sub.add_parser("stage", help="Validate and stage a candidate JSON file")
    p_stage.add_argument("--candidate", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "render-dify-inputs":
        print(json.dumps(build_dify_inputs(topic=args.topic, constraints=args.constraints, batch_id=args.batch_id), ensure_ascii=False, indent=2))
        return

    if args.cmd == "generate-dify":
        roots = []
        review_cfg = build_review_cfg_for_request(topic=args.topic, constraints=args.constraints)
        for candidate in generate_from_dify(topic=args.topic, constraints=args.constraints, batch_id=args.batch_id):
            findings = review_candidate(candidate, review_cfg=review_cfg)
            roots.append(str(write_candidate_package(candidate, findings)))
        print(json.dumps({"candidate_packages": roots}, ensure_ascii=False, indent=2))
        return

    if args.cmd == "validate":
        candidate, findings = validate_candidate_file(Path(args.candidate))
        print(
            json.dumps(
                {
                    "factor_key": candidate.factor_key,
                    "accepted_by_static_review": not any(x.severity == "error" for x in findings),
                    "findings": [vars(x) for x in findings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.cmd == "stage":
        root = stage_candidate_file(Path(args.candidate))
        print(json.dumps({"candidate_package": str(root)}, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    cli_main()
