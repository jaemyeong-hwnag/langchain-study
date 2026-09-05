"""DeepSeek 챗 모델 팩토리."""

import os
from functools import cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()

DEFAULT_MODEL = "deepseek-v4-flash"

def create_chat_model(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.7,
    thinking: bool = True,
) -> BaseChatModel:
    """기본값이 채워진 챗 모델을 생성한다.

    Args:
        model: 모델 식별자.
        api_key: 생략하면 환경변수 DEEPSEEK_API_KEY 를 사용한다.
        temperature: 샘플링 온도.
        thinking: False 면 모델의 내부 추론을 끈다. 단순 변환·추출 작업에서
            버려지는 추론 토큰이 사라져 훨씬 빨라진다.
    """
    if api_key is None:
        api_key = os.environ["DEEPSEEK_API_KEY"]
    llm = init_chat_model(model, api_key=api_key, temperature=temperature)
    if not thinking:
        llm = llm.bind(extra_body={"thinking": {"type": "disabled"}})
    return llm

@cache
def get_default_chat_model() -> BaseChatModel:
    """기본 챗 모델. 프로세스당 한 번만 생성된다."""
    return create_chat_model()
