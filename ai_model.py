from openai import OpenAI


class AiModel:
    def __init__(self, url: str, api_key: str, model: str, client=None):
        if client is None:
            self._client = OpenAI(
                base_url=url,
                api_key=api_key
            )
        else:
            self._client = OpenAI(
                base_url=url,
                api_key=api_key,
                http_client=client
            )
        self._model = model

    def comp(self, duty, prompt):
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": duty,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        return completion.choices[0].message.content