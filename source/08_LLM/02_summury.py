# ctrl+shift+p => 인터프리터 선택
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

def askGpt(prompt):
    "GPT에게 prompt요청 결과 반환"
    load_dotenv()
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role":"system", 
            "content":"당신은 한국어로 된 텍스트를 잘 요약하는 전문 어시스턴트입니다"
            },
            {"role":"user", "content":prompt}
        ]
    )
    return response.choices[0].message.content
# 기능 구현
def main():
    st.header("요약 프로그램")
    st.markdown("---")
    text = st.text_area("요약할 글을 입력하세요")
    if st.button("요약"):
        prompt = f"""your task is to summarize the text sentences in Korean language.
                Summarize in 1 lines. use the format of a bullet point.
                text : {text}"""
        result = askGpt(prompt=prompt)
        st.info(result)
if __name__ == "__main__":
    main()