from httpx import Client
from jsonrpcserver import method, serve, Result, Success

from ai_model import AiModel
from matcher import Matcher
from polling import MultiAiModelPolling
from config import config
from prompts import JOB_PARSE_PROMPT, JOB_PARSE_DUTY, RESUME_PARSE_PROMPT, RESUME_PARSE_DUTY

http_client = None

if config.ai_model.proxy is not None and config.ai_model.proxy != '':
    http_client = Client(proxy=config.ai_model.proxy)

ai_polling = MultiAiModelPolling(
    ai_model_instances=[
        AiModel(
            url=model.base_url,
            api_key=model.api_key,
            model=model.model_name,
            client=http_client
        )
        for model in config.ai_model.models],
    max_retries=3,
    retry_interval=1,
    health_check_interval=60
)

@method()
def job_parse(job_content) -> Result:
    prompt = JOB_PARSE_PROMPT + job_content
    json_content = ai_polling.comp(JOB_PARSE_DUTY, prompt).strip().split('```json')[1].split('```')[0]
    return Success(json_content)

@method()
def resume_parse(resume_content) -> Result:
    prompt = RESUME_PARSE_PROMPT + resume_content
    json_content = ai_polling.comp(RESUME_PARSE_DUTY, prompt).strip().split('```json')[1].split('```')[0]
    return Success(json_content)

@method()
def match(job_json, resume_json) -> Result:
    return Success(Matcher(job_json, resume_json).match())

if __name__ == '__main__':
    serve(
        port=config.json_rpc.port,
    )