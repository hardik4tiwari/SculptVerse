














from accelerate.logging import get_logger


logger = get_logger(__name__)


def configure_dynamo(config: dict):
    try:
        import torch._dynamo
        logger.debug(f'Configuring torch._dynamo.config with {config}')
        for k, v in config.items():
            if v is None:
                logger.debug(f'Skipping torch._dynamo.config.{k} with None')
                continue
            if hasattr(torch._dynamo.config, k):
                logger.warning(f'Overriding torch._dynamo.config.{k} from {getattr(torch._dynamo.config, k)} to {v}')
                setattr(torch._dynamo.config, k, v)
    except ImportError:
        logger.debug('torch._dynamo not found, skipping')
        pass
