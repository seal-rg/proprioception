import logging

logger = logging.getLogger(__name__)


def register() -> None:
    _name_gemma_top_k()
    from proprioception import probe_capture

    probe_capture.install()


def _name_gemma_top_k() -> None:
    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    except ImportError:
        return
    if not hasattr(Gemma4TextConfig, "num_experts_per_tok"):
        Gemma4TextConfig.num_experts_per_tok = property(lambda self: self.top_k_experts)
        logger.info("proprioception plugin: Gemma4TextConfig.num_experts_per_tok reads top_k_experts")
