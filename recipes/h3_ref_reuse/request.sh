#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail
request=${1:?Usage: request.sh REQUEST.json OUTPUT.mp4}
output=${2:?Usage: request.sh REQUEST.json OUTPUT.mp4}
# Preserve D045's synchronous multipart API, including reference upload order.
python - "$request" "$output" "${PORT:-19136}" <<'PY'
import json
import mimetypes
import os
import pathlib
import subprocess
import sys

request_path, output, port = sys.argv[1:]
request = json.loads(pathlib.Path(request_path).read_text(encoding="utf-8"))
fields = {
    "prompt": request["prompt"],
    "short_edge": str(request["target"]["short_edge"]),
    "aspect_ratio": request["target"]["aspect_ratio"],
    "fps": "24",
    "num_inference_steps": str(request["num_inference_steps"]),
    "flow_shift": str(request["flow_shift"]),
    "audio_flow_shift": str(request["audio_flow_shift"]),
    "seed": str(request["seed"]),
    "extra_params": json.dumps({"task": "ref2va", "duration": request["seconds"]}),
}
command = ["curl", "--fail-with-body", "--silent", "--show-error", "--max-time", "4000",
           "--output", output, "--write-out", "%{http_code} %{time_total}\n"]
for key, value in fields.items():
    command.extend(["--form-string", f"{key}={value}"])
input_root = pathlib.Path(os.environ.get("INPUT_DIR", "/inputs"))
for condition in request["conditions"]:
    path = input_root / condition["uri"]
    if not path.is_file():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    command.extend(["--form", f"input_references=@{path};type={mime}"])
command.append(f"http://127.0.0.1:{port}/v1/videos/sync")
with pathlib.Path(output + ".timing.txt").open("w", encoding="utf-8") as timing:
    subprocess.run(command, stdout=timing, check=True)
PY
