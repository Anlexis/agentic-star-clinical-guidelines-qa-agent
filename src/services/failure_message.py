"""AgentCore Platform v1.0"""

# Caller-facing sentences for a run that completes WITHOUT carrying out the
# request.
#
# A value the caller can correct ends the run here rather than terminating it:
# terminating would end the calling surface's turn and surface only an exception
# type, leaving the reason reachable solely from the audit trail. Completing with
# one of these sentences lets the caller fix the request and send it again on the
# same conversation.
#
# Each sentence names WHAT to correct and nothing else. It never echoes the
# rejected value, names an internal field path, or quotes a gate message — those
# stay in `error_log`, the internal audit channel.

EMPTY_INPUT = "No question was received. Send the question you want answered."
TOO_LONG = "The request is too long. Shorten it and send it again."
INVALID_VALUE = "A value in the request could not be accepted. Check it against the documented format."
INPUT_REJECTED = "The request could not be accepted. Check it against the documented format and send it again."

# Outcomes the caller cannot correct by rewording. These terminate.
PROCESSING_FAILED = "The request could not be processed. This is not something the request can fix."
