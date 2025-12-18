from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


@dataclass
class Job:
    id: str
    state: JobState
    created_at: float
    updated_at: float
    progress: float
    message: str | None
    error: str | None
    artifacts: dict[str, str]


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def create(self, job_id: str) -> Job:
        now = time.time()
        job = Job(
            id=job_id,
            state=JobState.queued,
            created_at=now,
            updated_at=now,
            progress=0.0,
            message=None,
            error=None,
            artifacts={},
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        state: JobState | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            if state is not None:
                job.state = state
            if progress is not None:
                job.progress = progress
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if artifacts is not None:
                job.artifacts.update(artifacts)

            job.updated_at = time.time()
            return job

    def as_dict(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id,
            "state": job.state.value,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
            "artifacts": job.artifacts,
        }
