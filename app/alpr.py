from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


_log = logging.getLogger("alpr")


_OPENOCR_ENGINES: dict[str, Any] = {}


def _openocr_module_diagnostics(mod) -> str:
    try:
        mod_file = getattr(mod, "__file__", None)
    except Exception:
        mod_file = None
    try:
        # `openocr-python` sets module-level `__dir__` to a string, which makes `dir(openocr)` crash.
        attrs = sorted(set(getattr(mod, "__dict__", {}).keys()))
    except Exception:
        attrs = []
    interesting = [
        a
        for a in attrs
        if a.lower()
        in {
            "infer",
            "openocr",
            "opendetector",
            "openrecognizer",
            "openocrdet",
            "openocrrec",
            "openocre2e",
        }
    ]
    return f"openocr_file={mod_file}; interesting_attrs={interesting}"


def _get_openocr_engine(*, prefer_gpu: bool):
    key = "gpu" if prefer_gpu else "cpu"
    if key in _OPENOCR_ENGINES:
        return _OPENOCR_ENGINES[key]

    try:
        from openocr import OpenOCR
    except Exception as e:
        raise RuntimeError(f"Failed to import OpenOCR from openocr-python: {type(e).__name__}: {e}")

    device = "cpu"
    if prefer_gpu:
        # OpenOCR ONNX backend uses onnxruntime. GPU requires onnxruntime-gpu + CUDA.
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                device = "gpu"
        except Exception:
            device = "cpu"

    try:
        engine = OpenOCR(backend="onnx", device=device, drop_score=0.0)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize OpenOCR engine: {type(e).__name__}: {e}"
        )

    _OPENOCR_ENGINES[key] = engine
    return engine


def _resolve_yolo_weights(model_path: str, *, cache_dir: Path | None) -> str:
    p = Path(model_path)
    if p.exists():
        return str(p)

    # Allow Hugging Face references:
    # - "org/repo:filename.pt"
    # - "org/repo/filename.pt" (root file)
    repo_id: str | None = None
    filename: str | None = None

    if ":" in model_path:
        left, right = model_path.split(":", 1)
        if "/" in left and right.strip().endswith(".pt"):
            repo_id = left.strip()
            filename = right.strip()
    else:
        parts = model_path.split("/")
        if len(parts) >= 3 and model_path.strip().endswith(".pt"):
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])

    if repo_id and filename:
        from huggingface_hub import hf_hub_download

        cache_dir = cache_dir or Path("models")
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir))
        return str(local_path)

    return model_path


def _safe_plate_text(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def _openocr_infer_image_path(img_path: str, *, prefer_gpu: bool) -> str:
    import openocr  # import here to avoid import cost when server starts

    # Validate that this is the expected `openocr-python` module.
    if not (hasattr(openocr, "OpenOCR") or hasattr(openocr, "OpenDetector") or hasattr(openocr, "OpenRecognizer")):
        diag = _openocr_module_diagnostics(openocr)
        _log.error("Invalid openocr module loaded: %s", diag)
        return (
            "OCR_ERROR:InvalidOpenOCRModule: "
            + diag
            + " | Fix: pip uninstall -y openocr && pip install -U openocr-python"
        )

    img = cv2.imread(img_path)
    if img is None:
        return ""

    try:
        engine = _get_openocr_engine(prefer_gpu=prefer_gpu)
        results, _time_dicts = engine(img_numpy=img)
    except Exception as e:
        _log.exception("OpenOCR engine call failed")
        return f"OCR_ERROR:{type(e).__name__}:{e}"

    # `results` is a list (one entry per input image). Each entry is a list of dicts like:
    # {"transcription": str, "points": [...], "score": float}
    try:
        if not results:
            return ""
        first = results[0] if isinstance(results, list) else results
        if not first:
            return ""
        texts: list[str] = []
        for item in first:
            if isinstance(item, dict) and isinstance(item.get("transcription"), str):
                texts.append(item["transcription"])
        return " ".join(texts)
    except Exception:
        return str(results)




def process_video(
    *,
    input_video_path: Path,
    output_video_path: Path,
    output_json_path: Path,
    model_path: str,
    conf: float,
    iou: float,
    use_gpu: bool,
    cache_dir: Path | None,
    progress_cb,
) -> None:
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video dimensions")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open output video writer")

    resolved_model_path = _resolve_yolo_weights(model_path, cache_dir=cache_dir)
    yolo = YOLO(resolved_model_path)

    yolo_device: str | int = "cpu"
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                yolo_device = 0
        except Exception:
            yolo_device = "cpu"

    results_out: dict[str, Any] = {
        "input": str(input_video_path),
        "model": resolved_model_path,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": [],
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="alpr_ocr_"))

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            dets = yolo.predict(frame, conf=conf, iou=iou, device=yolo_device, verbose=False)
            frame_items: list[dict[str, Any]] = []

            if dets and len(dets) > 0:
                r0 = dets[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is not None and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                    clss = boxes.cls.cpu().numpy() if boxes.cls is not None else None

                    for bi, b in enumerate(xyxy):
                        x1, y1, x2, y2 = [int(max(0, v)) for v in b]
                        x1 = min(x1, width - 1)
                        x2 = min(x2, width - 1)
                        y1 = min(y1, height - 1)
                        y2 = min(y2, height - 1)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        crop_path = tmp_dir / f"f{idx:08d}_b{bi:02d}.jpg"
                        cv2.imwrite(str(crop_path), crop)

                        text_raw = _openocr_infer_image_path(str(crop_path), prefer_gpu=use_gpu)
                        text = _safe_plate_text(text_raw)

                        score = float(confs[bi]) if confs is not None else None
                        cls_id = int(clss[bi]) if clss is not None else None

                        frame_items.append(
                            {
                                "box": [x1, y1, x2, y2],
                                "score": score,
                                "class_id": cls_id,
                                "text_raw": text_raw,
                                "text": text,
                            }
                        )

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = text if text else "PLATE"
                        if score is not None:
                            label = f"{label} {score:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

            results_out["frames"].append(
                {
                    "frame_index": idx,
                    "time_sec": float(idx / fps),
                    "detections": frame_items,
                }
            )

            writer.write(frame)
            idx += 1

            if frame_count > 0:
                progress_cb(min(0.999, idx / frame_count))
            else:
                progress_cb(0.0)

    finally:
        cap.release()
        writer.release()
        try:
            for p in tmp_dir.glob("*.jpg"):
                try:
                    os.remove(p)
                except Exception:
                    pass
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass
        except Exception:
            pass

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2)

    progress_cb(1.0)
