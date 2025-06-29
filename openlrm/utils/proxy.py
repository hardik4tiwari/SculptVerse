














import os

NO_PROXY = "OPENLRM_NO_DATA_PROXY" in os.environ

def no_proxy(func):
    """Decorator to disable proxy but then restore after the function call."""
    def wrapper(*args, **kwargs):

        http_proxy = os.environ.get('http_proxy')
        https_proxy = os.environ.get('https_proxy')
        HTTP_PROXY = os.environ.get('HTTP_PROXY')
        HTTPS_PROXY = os.environ.get('HTTPS_PROXY')
        all_proxy = os.environ.get('all_proxy')
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['all_proxy'] = ''
        try:
            return func(*args, **kwargs)
        finally:
            os.environ['http_proxy'] = http_proxy
            os.environ['https_proxy'] = https_proxy
            os.environ['HTTP_PROXY'] = HTTP_PROXY
            os.environ['HTTPS_PROXY'] = HTTPS_PROXY
            os.environ['all_proxy'] = all_proxy
    if NO_PROXY:
        return wrapper
    else:
        return func
