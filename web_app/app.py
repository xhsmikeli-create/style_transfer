import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

# Flask 1.x expects these names in jinja2; newer Jinja2 moved them to markupsafe.
try:
    import jinja2
    from markupsafe import Markup, escape

    if not hasattr(jinja2, "Markup"):
        jinja2.Markup = Markup
    if not hasattr(jinja2, "escape"):
        jinja2.escape = escape
except ImportError:
    pass

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_BOX_DIR = ROOT_DIR / "data_box"
INPUT_DIR = DATA_BOX_DIR / "input"
OUTPUT_DIR = DATA_BOX_DIR / "output"

CLASSIC_DIR = ROOT_DIR / "neural_style_transfer"
CLASSIC_STYLE_DIR = CLASSIC_DIR / "data" / "style-images"

FAST_DIR = ROOT_DIR / "fast_neural_style_transfer"
FAST_MODEL_DIR = FAST_DIR / "saved_models"

GAN_DIR = ROOT_DIR / "gan_style_generate"
GAN_MODEL_DIR = GAN_DIR / "model_weights"
GAN_RUNS_DIR = OUTPUT_DIR / "gan_runs"

COMPARE_DIR = ROOT_DIR / "comparison_runs" / "requested_existing_models_20260510" / "report_images"

MEDIA_ROOTS = {
    "input": INPUT_DIR,
    "output": OUTPUT_DIR,
    "classic-style": CLASSIC_STYLE_DIR,
    "compare": COMPARE_DIR,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODEL_SUFFIXES = {".pth", ".model"}
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

app = Flask(__name__)
app.jinja_options = app.jinja_options.copy()
app.jinja_options["extensions"] = [
    extension
    for extension in app.jinja_options.get("extensions", [])
    if extension not in {"jinja2.ext.autoescape", "jinja2.ext.with_"}
]
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

jobs = {}
jobs_lock = threading.Lock()
process_lock = threading.Lock()


def resolve_model_python():
    configured = Path(os.environ.get("PYTORCH_IMG2IMG_PYTHON", ""))
    if configured.is_file():
        return configured

    current = Path(sys.executable)
    if current.parent.name == "pytorch-img2img":
        return current

    if current.parent.name == "envs":
        candidate = current.parent / "pytorch-img2img" / "python.exe"
    else:
        candidate = current.parent / "envs" / "pytorch-img2img" / "python.exe"
    if candidate.is_file():
        return candidate

    for parent in current.parents:
        candidate = parent / "envs" / "pytorch-img2img" / "python.exe"
        if candidate.is_file():
            return candidate

    return current


MODEL_PYTHON = resolve_model_python()


def ensure_data_box():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GAN_RUNS_DIR.mkdir(parents=True, exist_ok=True)


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


def list_named_dirs(directory):
    if not directory.exists():
        return []
    return [
        {"name": path.name}
        for path in sorted(directory.iterdir(), key=lambda item: item.name.lower())
        if path.is_dir()
    ]


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
    candidate = directory / Path(str(name or "")).name
    if candidate.suffix.lower() not in suffixes or not candidate.is_file():
        raise ValueError(f"File does not exist or type is not supported: {name}")
    return candidate.name


def checked_dir_name(name, directory):
    candidate = directory / Path(str(name or "")).name
    if not candidate.is_dir():
        raise ValueError(f"Directory does not exist: {name}")
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


def validate_uploaded_image(upload):
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError("Only jpg, jpeg, and png images are supported.")

    try:
        with Image.open(upload.stream) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc
    finally:
        upload.stream.seek(0)


def save_uploaded_image(upload):
    validate_uploaded_image(upload)
    ensure_data_box()

    safe_name = secure_filename(upload.filename or "")
    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem or "image"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"
    destination = INPUT_DIR / filename
    upload.save(destination)
    return destination


def clear_input_dir():
    ensure_data_box()
    removed = 0
    for item in INPUT_DIR.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            item.unlink()
            removed += 1
    return removed


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
    append_log(job_id, "Job queued. Waiting for the worker...")

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
                update_job(job_id, status="error", error=f"Command exited with code {exit_code}")
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
        for item in directory.glob("*")
        if item.is_file()
        and item.suffix.lower() in IMAGE_SUFFIXES
        and item.stat().st_mtime >= since_timestamp - 2
    ]
    if not images:
        images = [
            item
            for item in directory.glob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ]
    if not images:
        raise FileNotFoundError(f"No output image found in {directory}")
    return max(images, key=lambda item: item.stat().st_mtime)


def copy_gan_result(raw_image_dir, output_path, temp_input_dir):
    fake_images = sorted(raw_image_dir.glob("*_fake.png"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not fake_images:
        fake_images = sorted(raw_image_dir.glob("*fake*.png"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not fake_images:
        raise FileNotFoundError(f"No GAN fake image found in {raw_image_dir}")

    shutil.copy2(fake_images[0], output_path)
    shutil.rmtree(temp_input_dir, ignore_errors=True)
    return output_path


@app.route("/", methods=["GET"])
def index():
    ensure_data_box()
    return render_template("index.html")


@app.route("/compare", methods=["GET"])
def compare():
    return render_template("compare.html")


@app.route("/api/assets", methods=["GET"])
def assets():
    ensure_data_box()
    input_images = list_named_files(INPUT_DIR, IMAGE_SUFFIXES, "input")
    return jsonify({
        "folders": {
            "input": "data_box/input",
            "output": "data_box/output",
        },
        "inputs": input_images,
        "classic": {
            "contents": input_images,
            "styles": input_images,
        },
        "fast": {
            "contents": input_images,
            "models": list_named_files(FAST_MODEL_DIR, MODEL_SUFFIXES),
        },
        "gan": {
            "contents": input_images,
            "models": list_named_dirs(GAN_MODEL_DIR),
        },
    })


@app.route("/api/upload", methods=["POST"])
def upload_image():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "No image file was provided."}), 400

    try:
        saved_path = save_uploaded_image(upload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "file": {
            "name": saved_path.name,
            "url": media_url(saved_path),
        }
    })


@app.route("/api/input/clean", methods=["POST"])
def clean_input():
    removed = clear_input_dir()
    return jsonify({"removed": removed})


@app.route("/api/comparison", methods=["GET"])
def comparison_assets():
    cases = [
        {
            "id": "mountain",
            "label": "Mountain",
            "description": "Sky, snow, forest, and foreground trees for natural scene comparison.",
        },
        {
            "id": "golden_gate",
            "label": "Golden Gate",
            "description": "Bridge towers, cables, and water structure for geometry comparison.",
        },
        {
            "id": "lion",
            "label": "Lion",
            "description": "Clear subject and dense fur texture for detail comparison.",
        },
    ]
    variants = [
        ("original", "Original", "{id}.jpg"),
        ("real", "GAN Input", "{id}_real.png"),
        ("fastUkiyoe", "Fast Ukiyoe", "{id}_fast_ukiyoe.jpg"),
        ("ganVanGogh", "GAN Van Gogh", "{id}_fake1.png"),
        ("ganUkiyoe", "GAN Ukiyoe", "{id}_fake2.png"),
    ]

    payload = []
    for item in cases:
        images = {}
        for key, label, pattern in variants:
            path = COMPARE_DIR / pattern.format(id=item["id"])
            images[key] = {
                "label": label,
                "name": path.name,
                "url": media_url(path) if path.is_file() else None,
            }
        payload.append({**item, "images": images})
    return jsonify({"cases": payload})


@app.route("/api/run/classic", methods=["POST"])
def run_classic():
    data = request.get_json(force=True)
    content_name = checked_name(data.get("content"), INPUT_DIR, IMAGE_SUFFIXES)
    style_name = checked_name(data.get("style"), INPUT_DIR, IMAGE_SUFFIXES)

    height = checked_int(data.get("height"), 400, 128, 1024)
    iterations = checked_int(data.get("iterations"), 500, 1, 5000)
    content_weight = checked_float(data.get("contentWeight"), 1e5)
    style_weight = checked_float(data.get("styleWeight"), 3e4)
    tv_weight = checked_float(data.get("tvWeight"), 1e-2)
    init_method = data.get("initMethod") if data.get("initMethod") in {"content", "style", "random"} else "content"

    ensure_data_box()
    classic_config = {
        "content_img_name": content_name,
        "style_img_name": style_name,
        "content_images_dir": str(INPUT_DIR),
        "style_images_dir": str(INPUT_DIR),
        "output_img_dir": str(OUTPUT_DIR),
        "height": height,
        "max_iterations_adam": iterations,
        "tv_weight": tv_weight,
        "content_weight": content_weight,
        "style_weight": style_weight,
        "init_method": init_method,
        "saving_freq": -1,
        "img_format": (4, ".jpg"),
    }
    runner = (
        "from neural_style_transfer import neural_style_transfer\n"
        f"config = {classic_config!r}\n"
        "neural_style_transfer(config)\n"
    )
    command = [str(MODEL_PYTHON), "-c", runner]

    started_at = time.time()
    output_dir = OUTPUT_DIR / f"combined_{Path(content_name).stem}_{Path(style_name).stem}"
    job_id = create_job("classic", command)
    thread = threading.Thread(
        target=run_process,
        args=(job_id, command, CLASSIC_DIR, lambda: latest_image(output_dir, started_at)),
        daemon=True,
    )
    thread.start()
    return jsonify({"jobId": job_id})


@app.route("/api/run/fast", methods=["POST"])
def run_fast():
    data = request.get_json(force=True)
    content_name = checked_name(data.get("content"), INPUT_DIR, IMAGE_SUFFIXES)
    model_name = checked_name(data.get("model"), FAST_MODEL_DIR, MODEL_SUFFIXES)

    ensure_data_box()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"fast_{Path(content_name).stem}_{Path(model_name).stem}_{timestamp}.jpg"

    fast_config = {
        "content_image": str(INPUT_DIR / content_name),
        "content_scale": None,
        "output_image": str(output_path),
        "model": str(FAST_MODEL_DIR / model_name),
        "accel": False,
    }
    runner = (
        "import sys\n"
        "from argparse import Namespace\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path('neural_style').resolve()))\n"
        "from neural_style import stylize\n"
        f"stylize(Namespace(**{fast_config!r}))\n"
    )
    command = [str(MODEL_PYTHON), "-c", runner]

    job_id = create_job("fast", command)
    thread = threading.Thread(
        target=run_process,
        args=(job_id, command, FAST_DIR, lambda: output_path),
        daemon=True,
    )
    thread.start()
    return jsonify({"jobId": job_id})


@app.route("/api/run/gan", methods=["POST"])
def run_gan():
    data = request.get_json(force=True)
    content_name = checked_name(data.get("content"), INPUT_DIR, IMAGE_SUFFIXES)
    model_name = checked_dir_name(data.get("model"), GAN_MODEL_DIR)

    checkpoint = GAN_MODEL_DIR / model_name / "latest_net_G.pth"
    if not checkpoint.is_file():
        return jsonify({"error": f"Missing GAN checkpoint: {checkpoint}"}), 400

    ensure_data_box()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_token = uuid.uuid4().hex[:10]
    temp_input_dir = OUTPUT_DIR / f"_gan_input_{job_token}"
    temp_input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT_DIR / content_name, temp_input_dir / content_name)

    raw_results_dir = GAN_RUNS_DIR / job_token
    raw_image_dir = raw_results_dir / model_name / "test_latest" / "images"
    output_path = OUTPUT_DIR / f"gan_{Path(content_name).stem}_{model_name}_{timestamp}.png"

    command = [
        str(MODEL_PYTHON),
        "generate.py",
        "--dataroot", str(temp_input_dir),
        "--name", model_name,
        "--checkpoints_dir", str(GAN_MODEL_DIR),
        "--results_dir", str(raw_results_dir),
        "--model", "test",
        "--dataset_mode", "single",
        "--num_test", "1",
        "--preprocess", "none",
        "--no_dropout",
        "--eval",
    ]

    job_id = create_job("gan", command)
    thread = threading.Thread(
        target=run_process,
        args=(job_id, command, GAN_DIR, lambda: copy_gan_result(raw_image_dir, output_path, temp_input_dir)),
        daemon=True,
    )
    thread.start()
    return jsonify({"jobId": job_id})


@app.route("/api/jobs/<job_id>", methods=["GET"])
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job)


@app.route("/media/<key>/<path:relative_path>", methods=["GET"])
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
    ensure_data_box()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
