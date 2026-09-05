import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableBranch

from init_chat_model import get_default_chat_model

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template("{topic}에 대해 간단히 설명해주세요.")

llm = get_default_chat_model()
output_parser = StrOutputParser()

### 체인 구성 (프롬프트 | LLM)
# chain = prompt | llm
#
# response = chain.invoke({"topic": "인공지능"})
# print(response.id)
# print(response.content)


### batch - 여러 입력 동시 처리
# chain = prompt | llm | StrOutputParser()
#
# results = chain.batch([
#     {"topic": "AI"},
#     {"topic": "ML"},
#     {"topic": "DL"}
# ])
#
# # stream - 실시간 출력
# for chunk in chain.stream({"topic": "AI"}):
#     print(chunk, end="", flush=True)
# # stream - 실시간 출력
# for chunk in chain.stream({"topic": "DL"}):
#     print(chunk, end="", flush=True)

### invoke - 단건 요청
# response = llm.invoke("지구의 자전 주기는?")
# print(response.id)
# print(response.content)

###
"""
다음 코드에서 prompt와 LLM 모델을 연결하여 chain을 구성하고, 이 chain을 사용하여 입력된 질문 "지구의 자전 주기는?"에 대한 답변을 생성하는 과정을 구현합니다. prompt 객체에 {"input": "지구의 자전 주기는?"} 라는 입력 값이 주어졌을 때, <Question>: {input} 부분의 {input} 위치로 "지구의 자전 주기는?" 값이 전달되어 질문에 대한 프롬프트를 완성합니다 .완성된 프롬프트는 그 후 LLM에 전달되어, 모델이 입력된 질문에 대한 답변을 생성하게 됩니다. 모델의 답변은 인공지능 모델의 메시지를 나타내는 AIMessage 객체로 제공됩니다.
"""
# prompt = ChatPromptTemplate(input_variables=['input'], messages=[HumanMessagePromptTemplate(
#     prompt=PromptTemplate(input_variables=['input'],
#                           template='You are an expert in astronomy. Answer the question. <Question>: {input}'))])
#
# # chain 연결 (LCEL)
# chain = prompt | llm | output_parser
#
# # chain 호출
# response = chain.invoke({"input": "지구의 자전 주기는?"})
# print(response)

# ### 기본 순차 체인
#
# # 1단계: 한국어를 영어로 번역
# translate_prompt = ChatPromptTemplate.from_template(
#     "다음 한국어를 영어로 번역하세요: {korean_word}"
# )
#
# # 2단계: 영어 단어 설명
# explain_prompt = ChatPromptTemplate.from_template(
#     "다음 영어 단어를 한국어로 자세히 설명하세요: {english_word}"
# )
#
# # 체인 1: 번역
# chain1 = translate_prompt | llm | StrOutputParser()
#
# # 체인 2: 번역 결과를 입력으로 사용
# chain2 = (
#     {"english_word": chain1}
#     | explain_prompt
#     | llm
#     | StrOutputParser()
# )
#
# # 실행
# result = chain2.invoke({"korean_word": "인공지능"})
# print(result)

# # 3단계 처리: 주제 분석 → 개요 작성 → 본문 작성
# # 너무 느림
# analyze_prompt = ChatPromptTemplate.from_template(
#     "다음 주제의 핵심 키워드 3개를 추출하세요: {topic}"
# )
#
# outline_prompt = ChatPromptTemplate.from_template(
#     """다음 키워드를 바탕으로 글의 개요를 작성하세요:
# 키워드: {keywords}
# 원본 주제: {topic}"""
# )
#
# content_prompt = ChatPromptTemplate.from_template(
#     """다음 개요를 바탕으로 300자 내외의 글을 작성하세요:
# 개요: {outline}"""
# )
#
# # 체인 구성
# chain = (
#     {
#         "topic": RunnablePassthrough()
#     }
#     | RunnablePassthrough.assign(
#         keywords=analyze_prompt | llm | StrOutputParser()
#     )
#     | RunnablePassthrough.assign(
#         outline=outline_prompt | llm | StrOutputParser()
#     )
#     | content_prompt
#     | llm
#     | StrOutputParser()
# )
#
# result = chain.invoke("기후 변화와 지속 가능한 발전")
# print(result)

# 3단계 처리: 주제 분석 → 개요 작성 → 본문 작성
# 성능 개선(스트리밍) - 똑같음
# analyze_prompt = ChatPromptTemplate.from_template(
#     "다음 주제의 핵심 키워드 3개를 추출하세요: {topic}"
# )
#
# outline_prompt = ChatPromptTemplate.from_template(
#     """다음 키워드를 바탕으로 글의 개요를 작성하세요:
# 키워드: {keywords}
# 원본 주제: {topic}"""
# )
#
# content_prompt = ChatPromptTemplate.from_template(
#     """다음 개요를 바탕으로 300자 내외의 글을 작성하세요:
# 개요: {outline}"""
# )
#
# # 체인 구성
# chain = (
#     {
#         "topic": RunnablePassthrough()
#     }
#     | RunnablePassthrough.assign(
#         keywords=analyze_prompt | llm | StrOutputParser()
#     )
#     | RunnablePassthrough.assign(
#         outline=outline_prompt | llm | StrOutputParser()
#     )
#     | content_prompt
#     | llm
#     | StrOutputParser()
# )
#
# # invoke -> stream 으로 변경. 마지막 단계의 토큰이 생성되는 즉시 흘러나온다.
# start = time.perf_counter()
# first = None
#
# for chunk in chain.stream("기후 변화와 지속 가능한 발전"):
#     if first is None:
#         first = time.perf_counter() - start
#     print(chunk, end="", flush=True)
#
# print(f"\n\n첫 출력까지: {first:.2f}s / 전체: {time.perf_counter() - start:.2f}s")

# 병렬 체인 실행
# 세 가지 관점에서 동시 분석
# positive_prompt = ChatPromptTemplate.from_template(
#     "{topic}의 긍정적인 측면 3가지를 설명하세요."
# )
#
# negative_prompt = ChatPromptTemplate.from_template(
#     "{topic}의 부정적인 측면 3가지를 설명하세요."
# )
#
# neutral_prompt = ChatPromptTemplate.from_template(
#     "{topic}에 대한 객관적인 현황을 설명하세요."
# )
#
# # 병렬 체인 구성
# parallel_chain = RunnableParallel(
#     positive=positive_prompt | llm | StrOutputParser(),
#     negative=negative_prompt | llm | StrOutputParser(),
#     neutral=neutral_prompt | llm | StrOutputParser(),
# )
#
# # 실행 (세 체인이 동시에 실행됨)
# results = parallel_chain.invoke({"topic": "원격 근무"})
#
# print("=== 긍정적 측면 ===")
# print(results["positive"])
# print("\n=== 부정적 측면 ===")
# print(results["negative"])
# print("\n=== 객관적 현황 ===")
# print(results["neutral"])

# 조건부 분기
# 언어별 다른 프롬프트
korean_prompt = ChatPromptTemplate.from_template(
    "다음 한국어 질문에 한국어로 답변하세요: {question}"
)

english_prompt = ChatPromptTemplate.from_template(
    "Answer the following question in English: {question}"
)

default_prompt = ChatPromptTemplate.from_template(
    "Please answer: {question}"
)

# 언어 감지 함수
def detect_language(input_dict):
    question = input_dict.get("question", "")
    # 간단한 한글 감지
    if any('\uac00' <= char <= '\ud7a3' for char in question):
        return "korean"
    return "english"

# 조건부 분기
branch_chain = RunnableBranch(
    # (조건 함수, 실행할 체인) 튜플 목록
    (lambda x: detect_language(x) == "korean", korean_prompt | llm | StrOutputParser()),
    (lambda x: detect_language(x) == "english", english_prompt | llm | StrOutputParser()),
    # 기본값 (조건에 맞지 않을 때)
    default_prompt | llm | StrOutputParser()
)

# 테스트
result_kr = branch_chain.invoke({"question": "파이썬이란 무엇인가요?"})
result_en = branch_chain.invoke({"question": "What is Python?"})

print("한국어 질문 결과:", result_kr)
print("영어 질문 결과:", result_en)