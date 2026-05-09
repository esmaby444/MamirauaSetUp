from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from .mamiraua_species import MAMIRAUA_SCIENTIFIC_NAMES

LOG = logging.getLogger(__name__)


@dataclass
class BirdResult:
    species_stats: dict[str, dict[str, float]]
    chunk_count: int
    active_chunk_count: int
    detection_count: int


class BirdDetector:
    def __init__(self, cfg: dict[str, Any], paths_cfg: dict[str, Any], regional_cfg: dict[str, Any] | None = None):
        self.model_path = Path(paths_cfg["model_path"])
        self.labels_path = Path(paths_cfg["labels_path"])
        self.allowlist_path = str(paths_cfg.get("species_allowlist_path", "")).strip()
        self.denylist_path = str(paths_cfg.get("species_denylist_path", "")).strip()

        self.sample_rate = int(cfg.get("sample_rate", 48000))
        self.chunk_seconds = float(cfg.get("chunk_seconds", 3.0))
        self.overlap_seconds = float(cfg.get("overlap_seconds", 0.0))
        self.min_conf = float(cfg.get("min_confidence", 0.2))
        self.top_k = int(cfg.get("top_k_per_chunk", 5))
        self.min_hits_per_window = int(cfg.get("min_hits_per_window", 1))
        self.sensitivity = float(cfg.get("sensitivity", 1.0))
        self.input_gain_db = float(cfg.get("input_gain_db", 0.0))
        self.input_gain_linear = 10.0 ** (self.input_gain_db / 20.0)
        self.legacy_processing = _to_bool(cfg.get("legacy_processing", True))
        self.legacy_force_sigmoid = _to_bool(cfg.get("legacy_force_sigmoid", True))
        self.legacy_center_audio = _to_bool(cfg.get("legacy_center_audio", True))
        self.legacy_use_mamiraua_whitelist = _to_bool(cfg.get("legacy_use_mamiraua_whitelist", True))

        self._labels: list[str] = []
        self._allowlist: set[str] = set()
        self._denylist: set[str] = set()
        self._effective_allowlist: set[str] = set()
        self._label_by_scientific: dict[str, set[str]] = {}
        self._mamiraua_labels: set[str] = set()
        self._mamiraua_scientific: set[str] = set(MAMIRAUA_SCIENTIFIC_NAMES)

        self.regional_cfg = regional_cfg or {}
        self.regional_enabled = bool(self.regional_cfg.get("enabled", False))
        self.regional_provider = str(self.regional_cfg.get("provider", "inaturalist")).strip().lower()
        self.regional_place_id = int(self.regional_cfg.get("place_id", 0) or 0)
        self.regional_radius_km = float(self.regional_cfg.get("radius_km", 1000.0))
        self.regional_min_observations = int(self.regional_cfg.get("min_observations", 1))
        self.regional_per_page = int(self.regional_cfg.get("per_page", 500))
        self.regional_max_pages = int(self.regional_cfg.get("max_pages", 20))
        self.regional_refresh_hours = float(self.regional_cfg.get("refresh_hours", 24))
        self.regional_movement_refresh_km = float(self.regional_cfg.get("movement_refresh_km", 25.0))
        self.regional_timeout_seconds = int(self.regional_cfg.get("request_timeout_seconds", 20))
        self.regional_quality_grade = str(self.regional_cfg.get("quality_grade", "any")).strip().lower()
        if self.regional_quality_grade in {"", "all"}:
            self.regional_quality_grade = "any"
        self.regional_verifiable_only = _to_bool(self.regional_cfg.get("verifiable_only", False))
        self.regional_cache_path = Path(
            str(self.regional_cfg.get("cache_path", "/opt/bird_station/state/regional_allowlist_cache.json"))
        )
        self.regional_include_non_bird_labels = {
            str(x).strip()
            for x in self.regional_cfg.get("include_non_bird_labels", [])
            if str(x).strip()
        }
        self._regional_last_refresh_utc: datetime | None = None
        self._regional_last_lat: float | None = None
        self._regional_last_lon: float | None = None

        self._interpreter = None
        self._input_audio = None
        self._input_metadata = None
        self._output = None
        self._chunk_samples = int(self.sample_rate * self.chunk_seconds)

    @property
    def loaded(self) -> bool:
        return self._interpreter is not None

    def ensure_loaded(self) -> None:
        if self.loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._labels = self._load_labels(self.labels_path)
        self._allowlist = self._load_allowlist(self.allowlist_path)
        self._denylist = self._load_denylist(self.denylist_path)
        self._effective_allowlist = set(self._allowlist)
        self._label_by_scientific = _build_label_index(self._labels)
        self._mamiraua_labels = _labels_for_scientific(self._label_by_scientific, self._mamiraua_scientific)
        if self.legacy_use_mamiraua_whitelist:
            LOG.info("Legacy Mamiraua whitelist active: %d labels", len(self._mamiraua_labels))
        self._load_regional_cache()

        try:
            import tflite_runtime.interpreter as tflite
        except Exception:
            try:
                from ai_edge_litert.interpreter import Interpreter as LiteRtInterpreter

                class _LiteRtModule:
                    Interpreter = LiteRtInterpreter

                tflite = _LiteRtModule()
            except Exception:
                try:
                    from tensorflow import lite as tflite  # type: ignore
                except Exception as exc:
                    raise RuntimeError(
                        "No LiteRT interpreter found. Install tflite-runtime (Py<=3.11) "
                        "or ai-edge-litert (Py>=3.12)."
                    ) from exc

        interpreter = tflite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()

        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if not inputs or not outputs:
            raise RuntimeError("Invalid TFLite model inputs/outputs")

        inputs_sorted = sorted(inputs, key=lambda d: _tensor_size(d.get("shape", [])), reverse=True)
        self._input_audio = inputs_sorted[0]
        self._input_metadata = None

        if len(inputs_sorted) > 1:
            candidate = inputs_sorted[-1]
            if _tensor_size(candidate.get("shape", [])) <= 16:
                self._input_metadata = candidate

        self._output = outputs[0]

        inferred_len = _infer_audio_samples(self._input_audio, self.sample_rate, self.chunk_seconds)
        if inferred_len > 0:
            self._chunk_samples = inferred_len

        self._interpreter = interpreter
        LOG.info(
            "Loaded model. audio_input=%s metadata_input=%s output=%s chunk_samples=%d labels=%d",
            self._input_audio.get("name"),
            self._input_metadata.get("name") if self._input_metadata else "none",
            self._output.get("name"),
            self._chunk_samples,
            len(self._labels),
        )

    def analyze_wav(self, wav_path: Path, lat: float | None, lon: float | None, when_utc: datetime) -> BirdResult:
        self.ensure_loaded()
        assert self._interpreter is not None
        assert self._input_audio is not None
        assert self._output is not None

        self._maybe_refresh_regional_allowlist(lat, lon)

        signal = _read_wav_mono_float(
            wav_path,
            expected_rate=self.sample_rate,
            gain_linear=self.input_gain_linear,
            center=self.legacy_center_audio,
        )
        if signal.size == 0:
            return BirdResult({}, 0, 0, 0)

        chunks = _split_signal(signal, self._chunk_samples, int(self.overlap_seconds * self.sample_rate))
        if not chunks:
            return BirdResult({}, 0, 0, 0)

        week = _week_index_1_to_48(when_utc)
        metadata = _build_metadata(lat, lon, week)

        species_stats: dict[str, dict[str, float]] = {}
        active_chunks = 0
        detections = 0
        chunk_hit_labels: list[set[str]] = []

        # Apply regional/explicit allowlist filtering BEFORE enforcing top-k.
        # This prevents non-local labels from consuming top-k slots and hiding
        # local candidates that have slightly lower scores.
        active_allowlist = self._effective_allowlist if self._effective_allowlist else self._allowlist

        for chunk in chunks:
            top = self._predict_chunk(
                chunk,
                metadata,
                candidate_allowlist=active_allowlist,
                candidate_denylist=self._denylist,
            )
            hits = [(label, score) for label, score in top if score >= self.min_conf]
            chunk_labels = {label for label, _ in hits}
            chunk_hit_labels.append(chunk_labels)

            if hits:
                active_chunks += 1

            for label, score in hits:
                entry = species_stats.setdefault(label, {"count": 0.0, "sum_conf": 0.0, "max_conf": 0.0})
                entry["count"] += 1.0
                entry["sum_conf"] += score
                entry["max_conf"] = max(entry["max_conf"], score)
                detections += 1

        # Optional denoising: require a species to appear in at least N chunks
        # within the window. This lets us use a lower confidence floor without
        # filling logs with one-off false positives.
        min_hits = max(1, int(self.min_hits_per_window))
        if min_hits > 1 and species_stats:
            kept_labels = {label for label, stats in species_stats.items() if int(stats.get("count", 0)) >= min_hits}
            species_stats = {label: stats for label, stats in species_stats.items() if label in kept_labels}
            detections = int(sum(int(stats.get("count", 0)) for stats in species_stats.values()))
            active_chunks = sum(1 for labels in chunk_hit_labels if labels & kept_labels)

        return BirdResult(
            species_stats=species_stats,
            chunk_count=len(chunks),
            active_chunk_count=active_chunks,
            detection_count=detections,
        )

    def _predict_chunk(
        self,
        chunk: np.ndarray,
        metadata: np.ndarray,
        candidate_allowlist: set[str] | None = None,
        candidate_denylist: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        assert self._interpreter is not None
        assert self._input_audio is not None
        assert self._output is not None

        audio_tensor = _reshape_audio_for_input(chunk, self._input_audio)
        _set_quantized_tensor(self._interpreter, self._input_audio, audio_tensor)

        if self._input_metadata is not None:
            metadata_tensor = _reshape_metadata_for_input(metadata, self._input_metadata)
            _set_quantized_tensor(self._interpreter, self._input_metadata, metadata_tensor)

        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output["index"])
        scores = _to_probability(
            raw,
            self.sensitivity,
            force_sigmoid=self.legacy_processing and self.legacy_force_sigmoid,
        ).ravel()

        labels = self._labels
        if not labels:
            labels = [f"class_{i}" for i in range(scores.shape[0])]
        if len(labels) < scores.shape[0]:
            labels.extend(f"class_{i}" for i in range(len(labels), scores.shape[0]))

        pairs = list(zip(labels[: scores.shape[0]], scores.tolist(), strict=False))
        combined_allowlist: set[str] = set()
        if candidate_allowlist:
            combined_allowlist.update(candidate_allowlist)
        if self.legacy_use_mamiraua_whitelist and self._mamiraua_labels:
            combined_allowlist.update(self._mamiraua_labels)
        if combined_allowlist:
            pairs = [(label, score) for label, score in pairs if label in combined_allowlist]
        if candidate_denylist:
            pairs = [(label, score) for label, score in pairs if label not in candidate_denylist]
        pairs.sort(key=lambda item: item[1], reverse=True)

        top = []
        max_items = len(pairs) if self.legacy_processing else self.top_k
        for label, score in pairs[:max_items]:
            if label in {"Human_Human", "Non-bird_Non-bird", "Noise_Noise"}:
                continue
            top.append((label, float(score)))

        return top

    def prime_filters(self, lat: float | None, lon: float | None) -> None:
        self.ensure_loaded()
        self._maybe_refresh_regional_allowlist(lat, lon)

    def _maybe_refresh_regional_allowlist(self, lat: float | None, lon: float | None) -> None:
        if not self.regional_enabled:
            return
        if self.regional_provider != "inaturalist":
            return
        if lat is None or lon is None:
            return

        now = datetime.now(UTC)
        if self._regional_last_refresh_utc is not None:
            age = now - self._regional_last_refresh_utc
            if age < timedelta(hours=max(0.25, self.regional_refresh_hours)):
                if self._regional_last_lat is not None and self._regional_last_lon is not None:
                    moved_km = _haversine_km(lat, lon, self._regional_last_lat, self._regional_last_lon)
                    if moved_km < max(0.5, self.regional_movement_refresh_km):
                        return

        scientific_names = self._fetch_regional_scientific_names(lat, lon)
        if not scientific_names:
            LOG.warning(
                "Regional allowlist refresh returned zero species "
                "(lat=%.5f lon=%.5f place_id=%d radius=%.1fkm min_obs=%d quality=%s verifiable_only=%s)",
                lat,
                lon,
                self.regional_place_id,
                self.regional_radius_km,
                self.regional_min_observations,
                self.regional_quality_grade,
                self.regional_verifiable_only,
            )
            return

        regional_labels: set[str] = set()
        for scientific in scientific_names:
            regional_labels.update(self._label_by_scientific.get(scientific, set()))

        regional_labels.update(self.regional_include_non_bird_labels)
        if self._allowlist:
            regional_labels.update(self._allowlist)

        if not regional_labels:
            LOG.warning("Regional filter fetched species but matched no model labels; skipping update")
            return

        self._effective_allowlist = regional_labels
        self._regional_last_refresh_utc = now
        self._regional_last_lat = float(lat)
        self._regional_last_lon = float(lon)
        self._save_regional_cache(scientific_names, regional_labels)
        LOG.info(
            "Regional allowlist updated: scientific=%d labels=%d center=(%.5f, %.5f) radius=%.1fkm",
            len(scientific_names),
            len(regional_labels),
            lat,
            lon,
            self.regional_radius_km,
        )

    def _fetch_regional_scientific_names(self, lat: float, lon: float) -> set[str]:
        base = "https://api.inaturalist.org/v1/observations/species_counts"
        per_page = max(20, min(200, self.regional_per_page))
        max_pages = max(1, min(50, self.regional_max_pages))
        timeout = max(5, min(60, self.regional_timeout_seconds))
        min_obs = max(1, self.regional_min_observations)
        names: set[str] = set()
        pages_with_results = 0

        try:
            for page in range(1, max_pages + 1):
                params = {
                    "lat": f"{lat:.6f}",
                    "lng": f"{lon:.6f}",
                    "radius": f"{max(1.0, self.regional_radius_km):.1f}",
                    "iconic_taxa": "Aves",
                    "hrank": "species",
                    "lrank": "species",
                    "per_page": str(per_page),
                    "page": str(page),
                }
                if self.regional_place_id > 0:
                    params["place_id"] = str(self.regional_place_id)
                if self.regional_verifiable_only:
                    params["verifiable"] = "true"
                if self.regional_quality_grade not in {"", "any"}:
                    params["quality_grade"] = self.regional_quality_grade
                url = f"{base}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers={"User-Agent": "bird-station/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                if not isinstance(payload, dict):
                    LOG.warning("Regional filter fetch got non-dict payload")
                    break
                if payload.get("error"):
                    LOG.warning("Regional filter fetch API error: %s", payload.get("error"))
                    break

                results = payload.get("results", [])
                if not isinstance(results, list) or not results:
                    break
                pages_with_results += 1

                for row in results:
                    if not isinstance(row, dict):
                        continue
                    obs_count = _to_int(
                        row.get("count"),
                        row.get("observations_count"),
                        row.get("observed_count"),
                    )
                    if obs_count < min_obs:
                        continue
                    taxon = row.get("taxon") if isinstance(row.get("taxon"), dict) else {}
                    scientific = str(taxon.get("name", "")).strip()
                    norm = _normalize_scientific(scientific)
                    if norm:
                        names.add(norm)

                total_results = _to_int(payload.get("total_results"))
                if total_results > 0 and page * per_page >= total_results:
                    break

        except urllib.error.URLError as exc:
            LOG.warning("Regional filter fetch failed (network): %s", exc)
            return set()
        except Exception as exc:
            LOG.warning("Regional filter fetch failed: %s", exc)
            return set()

        LOG.info(
            "Regional filter fetch complete: species=%d pages=%d place_id=%d radius=%.1fkm quality=%s verifiable_only=%s",
            len(names),
            pages_with_results,
            self.regional_place_id,
            self.regional_radius_km,
            self.regional_quality_grade,
            self.regional_verifiable_only,
        )
        return names

    def _load_regional_cache(self) -> None:
        if not self.regional_enabled:
            return
        try:
            if not self.regional_cache_path.exists():
                return
            data = json.loads(self.regional_cache_path.read_text(encoding="utf-8"))
            labels = data.get("labels", [])
            if isinstance(labels, list):
                cached = {str(x).strip() for x in labels if str(x).strip()}
                if cached:
                    self._effective_allowlist = set(self._allowlist) | cached
            ts = data.get("refreshed_at_utc")
            if isinstance(ts, str) and ts:
                self._regional_last_refresh_utc = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
            lat = data.get("lat")
            lon = data.get("lon")
            if lat is not None and lon is not None:
                self._regional_last_lat = float(lat)
                self._regional_last_lon = float(lon)
            if self._effective_allowlist:
                LOG.info("Loaded cached regional allowlist: %d labels", len(self._effective_allowlist))
        except Exception as exc:
            LOG.warning("Failed reading regional cache: %s", exc)

    def _save_regional_cache(self, scientific_names: set[str], labels: set[str]) -> None:
        try:
            self.regional_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "refreshed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "lat": self._regional_last_lat,
                "lon": self._regional_last_lon,
                "radius_km": self.regional_radius_km,
                "scientific_names": sorted(scientific_names),
                "labels": sorted(labels),
            }
            self.regional_cache_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception as exc:
            LOG.warning("Failed writing regional cache: %s", exc)

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        if not path.exists():
            LOG.warning("Labels file not found: %s", path)
            return []
        labels = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                labels.append(line)
        return labels

    @staticmethod
    def _load_allowlist(path: str) -> set[str]:
        if not path:
            return set()
        p = Path(path)
        if not p.exists():
            LOG.warning("Allowlist not found: %s", p)
            return set()
        allowed: set[str] = set()
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                name = line.strip()
                if name:
                    allowed.add(name)
        LOG.info("Loaded allowlist: %d species", len(allowed))
        return allowed

    @staticmethod
    def _load_denylist(path: str) -> set[str]:
        if not path:
            return set()
        p = Path(path)
        if not p.exists():
            LOG.warning("Denylist not found: %s", p)
            return set()
        denied: set[str] = set()
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                name = line.strip()
                if name:
                    denied.add(name)
        LOG.info("Loaded denylist: %d species", len(denied))
        return denied


def _tensor_size(shape: Any) -> int:
    total = 1
    try:
        for dim in shape:
            d = int(dim)
            total *= max(1, abs(d))
    except Exception:
        return 0
    return total


def _normalize_scientific(name: str) -> str:
    n = " ".join(str(name).strip().split())
    return n.lower()


def _build_label_index(labels: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for label in labels:
        raw = str(label).strip()
        if not raw:
            continue
        left = raw.split("_", 1)[0].strip()
        scientific = _normalize_scientific(left if left else raw)
        if scientific:
            out.setdefault(scientific, set()).add(raw)
    return out


def _labels_for_scientific(index: dict[str, set[str]], scientific_names: set[str]) -> set[str]:
    labels: set[str] = set()
    for scientific in scientific_names:
        labels.update(index.get(scientific, set()))
    return labels


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dp / 2) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _to_int(*values: Any) -> int:
    for value in values:
        try:
            if value is None:
                continue
            return int(value)
        except Exception:
            continue
    return 0


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _infer_audio_samples(input_detail: dict[str, Any], sample_rate: int, chunk_seconds: float) -> int:
    shape = [int(x) for x in input_detail.get("shape", [])]
    candidates = [dim for dim in shape if dim > 1000]
    if candidates:
        return max(candidates)
    return int(sample_rate * chunk_seconds)


def _reshape_audio_for_input(chunk: np.ndarray, input_detail: dict[str, Any]) -> np.ndarray:
    shape = [int(x) for x in input_detail.get("shape", [])]
    if not shape:
        return chunk.astype(np.float32)

    fixed = [1 if dim < 1 else dim for dim in shape]
    flat_size = 1
    for dim in fixed:
        flat_size *= dim

    flat = np.zeros((flat_size,), dtype=np.float32)
    n = min(flat_size, chunk.shape[0])
    flat[:n] = chunk[:n]
    return flat.reshape(fixed)


def _reshape_metadata_for_input(metadata: np.ndarray, input_detail: dict[str, Any]) -> np.ndarray:
    shape = [int(x) for x in input_detail.get("shape", [])]
    if not shape:
        return metadata.astype(np.float32)

    fixed = [1 if dim < 1 else dim for dim in shape]
    flat_size = 1
    for dim in fixed:
        flat_size *= dim

    src = metadata.astype(np.float32).ravel()
    flat = np.zeros((flat_size,), dtype=np.float32)
    n = min(flat_size, src.shape[0])
    flat[:n] = src[:n]
    return flat.reshape(fixed)


def _set_quantized_tensor(interpreter: Any, detail: dict[str, Any], value: np.ndarray) -> None:
    dtype = detail["dtype"]
    q = detail.get("quantization", (0.0, 0))
    scale, zero = float(q[0]), int(q[1])

    if dtype == np.float32:
        interpreter.set_tensor(detail["index"], value.astype(np.float32))
        return

    if scale > 0:
        quantized = np.round((value / scale) + zero)
    else:
        quantized = value

    if dtype == np.int8:
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
    elif dtype == np.uint8:
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    else:
        quantized = quantized.astype(dtype)

    interpreter.set_tensor(detail["index"], quantized)


def _to_probability(raw: np.ndarray, sensitivity: float, force_sigmoid: bool = False) -> np.ndarray:
    if force_sigmoid:
        s = float(max(0.25, min(3.0, sensitivity)))
        return 1.0 / (1.0 + np.exp(-s * raw.astype(np.float32)))
    # If model already outputs probabilities, keep as-is.
    if np.min(raw) >= 0.0 and np.max(raw) <= 1.0:
        return raw.astype(np.float32)
    # Otherwise apply sigmoid on logits.
    s = float(max(0.25, min(3.0, sensitivity)))
    return 1.0 / (1.0 + np.exp(-s * raw.astype(np.float32)))


def _read_wav_mono_float(path: Path, expected_rate: int, gain_linear: float = 1.0, center: bool = False) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width * 8}-bit")

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)

    if gain_linear > 1.0:
        pcm = np.clip(pcm * gain_linear, -1.0, 1.0)

    if center and pcm.size > 0:
        pcm = pcm - float(np.mean(pcm))
        pcm = np.clip(pcm, -1.0, 1.0)

    if rate != expected_rate:
        raise RuntimeError(
            f"Unexpected sample rate {rate}. Expected {expected_rate}. "
            "Set arecord sample_rate in config to match model input."
        )

    return pcm


def _split_signal(signal: np.ndarray, chunk_samples: int, overlap_samples: int) -> list[np.ndarray]:
    step = max(1, chunk_samples - overlap_samples)
    chunks: list[np.ndarray] = []
    for start in range(0, signal.shape[0], step):
        end = start + chunk_samples
        piece = signal[start:end]
        if piece.shape[0] < int(chunk_samples * 0.5):
            break
        if piece.shape[0] < chunk_samples:
            padded = np.zeros((chunk_samples,), dtype=np.float32)
            padded[: piece.shape[0]] = piece
            piece = padded
        chunks.append(piece)
    return chunks


def _week_index_1_to_48(when_utc: datetime) -> int:
    month = when_utc.month
    week_4x = ((month - 1) * 4) + max(1, min((when_utc.day - 1) // 7 + 1, 4))
    return max(1, min(48, week_4x))


def _build_metadata(lat: float | None, lon: float | None, week_1_48: int) -> np.ndarray:
    lat_val = -1.0 if lat is None else float(lat)
    lon_val = -1.0 if lon is None else float(lon)

    week_cos = math.cos(math.radians(float(week_1_48) * 7.5)) + 1.0 if 1 <= week_1_48 <= 48 else -1.0

    mask = np.ones((3,), dtype=np.float32)
    if lat is None or lon is None:
        mask[:] = 0.0
    if week_cos < 0:
        mask[2] = 0.0

    meta = np.array([lat_val, lon_val, week_cos], dtype=np.float32)
    return np.concatenate([meta, mask]).astype(np.float32)
