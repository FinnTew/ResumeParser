import yaml

class JsonRpcConfig:
    def __init__(self, port):
        self.port = port

class AiModelConfig:
    def __init__(self, proxy, models, polling):
        self.proxy = proxy
        self.models = models
        self.polling = polling

class Model:
    def __init__(self, model_name, base_url, api_key):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

class PollingConfig:
    def __init__(self, max_retries, retry_interval, health_check_interval):
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.health_check_interval = health_check_interval

class ConfigSingleton:
    _instance = None

    def __new__(cls, config_file='config.yaml'):
        if cls._instance is None:
            cls._instance = super(ConfigSingleton, cls).__new__(cls)
            cls._instance._initialize(config_file)
        return cls._instance

    def _initialize(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

            self.json_rpc = JsonRpcConfig(**config_data['json_rpc'])

            polling_data = config_data['ai_model']['polling']
            polling_config = PollingConfig(**polling_data)

            models_data = config_data['ai_model']['models']
            self.models = [Model(**model) for model in models_data]

            self.ai_model = AiModelConfig(config_data['ai_model']['proxy'], self.models, polling_config)


config = ConfigSingleton()