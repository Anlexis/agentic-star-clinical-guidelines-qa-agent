"""AgentCore Platform v1.0 — caller-facing progress events.

Sends non-terminal progress events so a caller watching the run sees the pipeline advance instead
of a silent wait. The emitter is resolved lazily and every failure is swallowed: progress reporting
is best-effort by contract and must never change the outcome of a run, and the module must import
cleanly in environments where the platform events package is absent.

Only non-terminal events are sent from here. Terminal delivery (success/failure) belongs to the
platform runner alone, which derives it from the invocation result.

Messages are static phase labels. Never pass caller content, secrets, or record values — a progress
event leaves the process and is not covered by the S-3 output gate.
"""


def emit_progress(message: str) -> None:
    """Report a pipeline phase to the caller. No-op when no emitter is bound or available.

    No completion percentage is sent: a run that reports 90% and then fails reads as a broken
    progress bar, and these phases carry no meaningful proportion of the work.
    """
    try:
        from shared.services.events import emitter
        from shared.services.events.types import EventType

        emitter().emit_event(event_type=EventType.PROGRESS_UPDATE, message=message, metadata={})
    except Exception:  # noqa: BLE001 — progress reporting must never affect the run
        return
