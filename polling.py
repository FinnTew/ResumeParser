import time

from ai_model import AiModel


class MultiAiModelPolling:
    def __init__(self, ai_model_instances, max_retries=5, retry_interval=2, health_check_interval=60):
        self._ai_model_instances = ai_model_instances
        self._max_retries = max_retries
        self._retry_interval = retry_interval
        self._health_check_interval = health_check_interval

        self._last_health_check = 0
        self._healthy_ai_models = ai_model_instances

    @staticmethod
    def health_check(model: AiModel) -> bool:
        try:
            test_response = model.comp("system", "health_check")
            return test_response is not None
        except Exception as e:
            print(f"Health check failed for {model}: {e}")
            return False

    def perform_health_check(self):
        print("Performing health check...")
        self._healthy_ai_models = [model for model in self._ai_model_instances if self.health_check(model)]
        if not self._healthy_ai_models:
            print("No healthy AiModel instances found!")
        else:
            print(f"Healthy AiModel instances: {len(self._healthy_ai_models)}")

    def comp(self, duty: str, prompt: str) -> str:
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            self.perform_health_check()
            self._last_health_check = current_time

        if not self._healthy_ai_models:
            raise Exception("No healthy AiModel instances available for polling.")

        for model in self._healthy_ai_models:
            attempt = 0
            while attempt < self._max_retries:
                try:
                    result = model.comp(duty, prompt)
                    if result:
                        print(f"AiModel {model} returned a valid response.")
                        return result
                except Exception as e:
                    print(f"Attempt {attempt + 1} for AiModel {model} failed: {e}")

                attempt += 1
                if attempt < self._max_retries:
                    time.sleep(self._retry_interval)

            print(f"AiModel {model} failed after {self._max_retries} attempts. Moving to the next instance.")

        raise Exception(f"All healthy AiModel instances failed to get a valid response after {self._max_retries} attempts each.")