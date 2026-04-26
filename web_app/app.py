import argparse
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from flask import Flask, abort, jsonify, render_template, request, send_from_directory


ROOT_DIR = Path(__file__).resolve().parents[1]

CLASSIC_DIR = ROOT_DIR / "neural_style_transfer"
CLASSIC_CONTENT_DIR = CLASSIC_DIR / "data" / "content-images"
CLASSIC_STYLE_DIR = CLASSIC_DIR / "data" / "style-images"
CLASSIC_OUTPUT_DIR = CLASSIC_DIR / "data" / "output-images"

FAST_DIR = ROOT_DIR / "fast_neural_style_transfer"
FAST_CONTENT_DIR = FAST_DIR / "images" / "content-images"
FAST_MODEL_DIR = FAST_DIR / "saved_models"
FAST_OUTPUT_DIR = FAST_DIR / "output"

MEDIA_ROOTS = {
    "classic-content": CLASSIC_CONTENT_DIR,
    "classic-style": CLASSIC_STYLE_DIR,
    "classic-output": CLASSIC_OUTPUT_DIR,
    "fast-content": FAST_CONTENT_DIR,
    "fast-output": FAST_OUTPUT_DIR,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODEL_SUFFIXES = {".pth", ".model"}

app = Flask(__name__)
jobs = {}
jobs_lock = threading.Lock()
process_lock = threading.Lock()


def list_named_files(directory, suffixes, media_key=None):
    if not directory.exists():
        return []

    files = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in suffixes:
            item = {"name": path.name}
            if media_key is not None:
                item["url"] = media_url(path)
            files.append(item)
    return files


def media_url(path):
    resolved = Path(path).resolve()
    for key, root in MEDIA_ROOTS.items():
        root = root.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return f"/media/{key}/{quote(relative.as_posix())}"
    return None


def checked_name(name, directory, suffixes):
    candidate = directory / Path(name).name
    if candidate.suffix.lower() not in suffixes or not candidate.is_file():
        raise ValueError(f"文件不存在或类型不支持: {name}")
    return candidate.name


def checked_int(value, default, minimum, maximum):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def checked_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def create_job(method, command):
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "method": method,
            "status": "queued",
            "command": " ".join(command),
            "logs": [],
            "result_url": None,
            "result_path": None,
            "error": None,
        }
    return job_id


def update_job(job_id, **changes):
    with jobs_lock:
        jobs[job_id].update(changes)


def append_log(job_id, line):
    with jobs_lock:
        logs = jobs[job_id]["logs"]
        logs.append(line.rstrip())
        del logs[:-120]


def run_process(job_id, command, cwd, result_finder):
    append_log(job_id, "任务已提交，等待 GPU 空闲。")

    with process_lock:
        update_job(job_id, status="running")
        append_log(job_id, f"$ {' '.join(command)}")

        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            assert process.stdout is not None
            for line in process.stdout:
                append_log(job_id, line)

            exit_code = process.wait()
            if exit_code != 0:
                update_job(job_id, status="error", error=f"命令退出码: {exit_code}")
                return

            result_path = result_finder()
            update_job(
                job_id,
                status="done",
                result_path=str(result_path),
                result_url=media_url(result_path),
            )
        except Exception as exc:
            update_job(job_id, status="error", error=str(exc))


def latest_image(directory, since_timestamp):
    images = [
        item
        for item in directory.glob("*.jpg")
        if item.is_file() and item.stat().st_mtime >= since_timestamp - 2
    ]
    if not images:
        images = [item for item in directory.glob("*.jpg") if item.is_file()]
    if not images:
        raise FileNotFoundError(f"没有找到输出图片: {directory}")
    return max(images, key=lambda item: item.stat().st_mtime)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/assets")
def assets():
    return jsonify({
        "classic": {
            "contents": list_named_files(CLASSIC_CONTENT_DIR, IMAGE_SUFFIXES, "classic-content"),
            "styles": list_named_files(CLASSIC_STYLE_DIR, IMAGE_SUFFIXES, "classic-style"),
        },
        "fast": {
            "contents": list_named_files(FAST_CONTENT_DIR, IMAGE_SUFFIXES, "fast-content"),
            "models": list_named_files(FAST_MODEL_DIR, MODEL_SUFFIXES),
        },
    })


@app.post("/api/run/classic")
def run_classic():
    data = request.get_json(force=True)
    content_name = checked_name(data.get("content"), CLASSIC_CONTENT_DIR, IMAGE_SUFFIXES)
    style_name = checked_name(data.get("style"), CLASSIC_STYLE_DIR, IMAGE_SUFFIXES)

    height = checked_int(data.get("height"), 400, 128, 1024)
    iterations = checked_int(data.get("iterations"), 500, 1, 5000)
    content_weight = checked_float(data.get("contentWeight"), 1e5)
    style_weight = checked_float(data.get("styleWeight"), 3e4)
    tv_weight = checked_float(data.get("tvWeight"), 1e-2)
    init_method = data.get("initMethod") if data.get("initMethod") in {"content", "style", "random"} else "content"

    command = [
        sys.executable,
        "neural_style_transfer.py",
        "--content_img_name", content_name,
        "--style_img_name", style_name,
        "--height", str(height),
        "--max_iterations_adam", str(iterations),
        "--tv_weight", str(tv_weight),
        "--content_weight", str(content_weight),
        "--style_weight", str(style_weight),
        "--init_method", init_method,
        "--saving_freq", "-1",
    ]

    started_at = time.time()
    output_dir = CLASSIC_OUTPUT_DIR / f"combined_{Path(content_name).stem}_{Path(style_name).stem}"
    job_id = create_job("classic", command)
    thread = threading.Thread(
        target=run_process,
        args=(job_id, command, CLASSIC_DIR, lambda: latest_image(output_dir, started_at)),
        daemon=True,
    )
    thread.start()
    return jsonify({"jobId": job_id})


@app.post("/api/run/fast")
def run_fast():
    data = request.get_json(force=True)
    content_name = checked_name(data.get("content"), FAST_CONTENT_DIR, IMAGE_SUFFIXES)
    model_name = checked_name(data.get("model"), FAST_MODEL_DIR, MODEL_SUFFIXES)

    FAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = FAST_OUTPUT_DIR / f"web_{Path(content_name).stem}_{Path(model_name).stem}_{timestamp}.jpg"

    command = [
        sys.executable,
        "neural_style/neural_style.py",
        "eval",
        "--content-image", str(Path("images") / "content-images" / content_name),
        "--output-image", str(Path("output") / output_path.name),
        "--model", str(Path("saved_models") / model_name),
        "--accel",
    ]

    job_id = create_job("fast", command)
    thread = threading.Thread(
        target=run_process,
        args=(job_id, command, FAST_DIR, lambda: output_path),
        daemon=True,
    )
    thread.start()
    return jsonify({"jobId": job_id})


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job)


@app.get("/media/<key>/<path:relative_path>")
def media(key, relative_path):
    root = MEDIA_ROOTS.get(key)
    if root is None:
        abort(404)
    safe_path = Path(relative_path)
    if safe_path.is_absolute() or ".." in safe_path.parts:
        abort(404)
    return send_from_directory(root, safe_path.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
