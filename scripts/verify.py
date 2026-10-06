"""Run existing commands and save their exit codes, logs and measured evidence.

This only orchestrates the existing CLI; it contains no training/inference logic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/verification")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Verification output must be under this repository's artifacts directory.")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OMP_NUM_THREADS="1")
    cli = ["palmistry_strict_auto_onefile.py"]

    def run(name, command, expected=0):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, text=True,
                                encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        (output / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        records.append({"name": name, "argv": [sys.executable, *command], "exit_code": result.returncode,
                        "expected_exit_code": expected, "seconds": time.perf_counter()-start})
        (output / "commands.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"{name}: exit={result.returncode} ({records[-1]['seconds']:.3f}s)", flush=True)
        if result.returncode != expected:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")

    run("compile", ["-m", "compileall", "-q", *[p.name for p in ROOT.glob("*.py")], "tests", "scripts", "palm_keypoints"])
    run("imports", ["-c", "import palmistry_strict_auto_onefile, tien_xu_ly, giao_dien_ui, prototype_common, prototype_fixture; print('all imports OK')"])
    run("http-demo", ["-c", "import threading,urllib.request; import giao_dien_ui as ui; from werkzeug.serving import make_server; server=make_server(ui.HOST,0,ui.app); worker=threading.Thread(target=server.serve_forever,daemon=True); worker.start(); response=urllib.request.urlopen('http://127.0.0.1:'+str(server.server_port)+'/api/status'); assert response.status==200; print('loopback HTTP',response.status); server.shutdown(); worker.join(); server.server_close()"])
    run("help", cli + ["--help"])
    run("doctor", cli + ["doctor"])
    run("dependencies", ["-m", "pip", "check"])
    run("freeze", ["-m", "pip", "freeze"])
    run("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])
    fixture = output / "fixture"
    run("fixture", cli + ["fixture", "--output_dir", str(fixture)])
    run("preprocess", cli + ["single-mask", "--image", str(fixture / "input.png"), "--output_dir", str(output / "preprocessed"), "--out_size", "128", "--min_short_side", "128"])
    predict = cli + ["predict", "--image", str(fixture / "input.png"), "--out_size", "128", "--min_short_side", "128", "--min_image_quality", "0", "--device", "cpu"]
    run("train", cli + ["--seed", "42", "train", "--data_root", str(fixture), "--labels_csv", str(fixture / "labels.csv"),
        "--out_dir", str(output / "model-smoke"), "--input_size", "32", "--epochs", "1", "--batch_size", "1", "--grad_accum", "2", "--model_size", "tiny", "--device", "cpu", "--resume", "none"])
    checkpoint = output / "model-smoke/checkpoints/best.npz"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for mode in ("classical", "model"):
        options = ["--allow-classical-fallback"] if mode == "classical" else ["--checkpoint", str(checkpoint), "--checkpoint-sha256", digest]
        folder = output / mode
        run(mode + "-predict", predict + options + ["--out_json", str(folder / "result.json"), "--out_mask", str(folder / "mask.png"), "--out_overlay", str(folder / "overlay.png")])
        run(mode + "-evaluate", cli + ["evaluate", "--prediction", str(folder / "mask.png"), "--target", str(fixture / "expected_mask.png"),
            "--dataset-name", "geometric-v1", "--split", "fixture-only", "--inference-source", "trained model" if mode == "model" else "classical CV fallback", "--output", str(folder / "evaluation.json")])
    # Real subprocess IPC with an offline fixture, without opening a camera.
    config = json.loads((ROOT / "prototype_config.json").read_text(encoding="utf-8"))
    config.update(out_size=128, min_image_quality=0)
    config_path = output / "fixture-ui-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    code = ("import os,json,cv2; from pathlib import Path; "
            + "os.environ['PALM_CONFIG']=" + repr(str(config_path)) + "; "
            + "os.environ['PALM_CHECKPOINT']=" + repr(str(checkpoint)) + "; "
            + "os.environ['PALM_CHECKPOINT_SHA256']=" + repr(str(digest)) + "; "
            + "import giao_dien_ui as ui; e=ui.LiveEngine(); assert e.ready; "
            + "frame=cv2.imread(" + repr(str(fixture / "input.png")) + "); "
            + "mask,overlay,data,metrics,reading=e.run_predict(frame); "
            + "assert mask.shape==overlay.shape[:2]; "
            + "Path(" + repr(str(output / "ui-runtime.json")) + ").write_text(json.dumps({'shape':list(mask.shape),'elapsed_seconds':data['elapsed_seconds'],'subprocess_end_to_end_seconds':data['subprocess_end_to_end_seconds']})); print('offline UI subprocess OK')")
    run("ui-offline", ["-c", code])
    run("bad-image", cli + ["predict", "--image", str(output / "not-present.png"), "--allow-classical-fallback", "--out_json", str(output / "bad-image.json"), "--out_mask", str(output / "bad-mask.png"), "--out_overlay", str(output / "bad-overlay.png")], expected=2)
    # Measure process creation separately from inference (three local warm-cache samples).
    startup = []
    for _ in range(3):
        start = time.perf_counter()
        subprocess.run([sys.executable, "-c", "pass"], check=True, cwd=ROOT, env=env, timeout=30)
        startup.append(time.perf_counter() - start)
    summary = {"date_utc": time.strftime("%Y-%m-%d", time.gmtime()), "python": sys.version,
               "process_startup_seconds": startup, "process_startup_median_seconds": statistics.median(startup),
               "fixture": json.loads((fixture / "expected.json").read_text(encoding="utf-8")),
               "commands": records, "metrics": {}}
    for mode in ("classical", "model"):
        result_path = output / mode / "evaluation.json"
        if result_path.exists():
            summary["metrics"][mode] = json.loads(result_path.read_text(encoding="utf-8"))
    (output / "evidence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
