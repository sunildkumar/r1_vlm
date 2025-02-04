import logging

from yaml import safe_load

from reasoncv.opbase import RcvOpFactory, RcvOpRequest

logger = logging.getLogger(__name__)


class RcvOpParser:
    """
    Parses the `<op>` tag and the YAML-formatted string that describes the operation.

    If `quiet` is True, then we generally ignore problems, and just return None, while
    penalizing the reward.  Use `quiet` when training and responding to stupid LLM
    responses.
    """

    START = "<op>"
    END = "</op>"

    def __init__(self, quiet: bool = False):
        self.reward = 0
        self.quiet = quiet

    def _problem(self, message: str, exc: Exception | None = None):
        """
        Logs the warning message. If an exception is provided,
        include its details by building the appropriate exception tuple.
        """
        self.reward -= 1
        if not self.quiet:
            if exc:
                logger.warning(message, exc_info=(type(exc), exc, exc.__traceback__))
            else:
                logger.warning(message)
            raise ValueError(message) from exc

    def parse(self, thinking: str) -> list[RcvOpRequest]:
        """
        Parses the `<op>` tag and the YAML-formatted string that describes the operation.
        """
        # Loop through the thinking string and find the <op> tag
        out = []
        while self.START in thinking:
            # Find the <op> tag
            start = thinking.find(self.START)
            if start == -1:
                break
            end = thinking.find(self.END)
            if end == -1:
                self._problem("No closing </op> tag found.")
                break

            op_contents = thinking[start+len(self.START):end]
            op_request = self.parse_op_str(op_contents)
            if op_request:
                out.append(op_request)

            # remove the <op> and </op> tags and the YAML-formatted string from the thinking
            thinking = thinking[end+len(self.END):]

        return out

    def parse_op_str(self, op_contents: str) -> RcvOpRequest | None:
        """
        Parses the YAML-formatted string that describes the operation.
        """
        try:
            op_doc = safe_load(op_contents)
            return RcvOpFactory.from_doc(op_doc)
        except Exception as e:
            self._problem(f"Error parsing operation: {e}", exc=e)
            return None
