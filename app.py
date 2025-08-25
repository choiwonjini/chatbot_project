import streamlit as st
from 분리수GO_chatbot import get_rag_answer

# 1. 페이지 제목 설정
st.title("🤖 분리수GO 챗봇 🚮")

# 2. 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 이전 대화 내용 모두 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 사용자 입력 받기
if prompt := st.chat_input("무엇이 궁금하신가요? (예: 페트병 버리는 법)"):
    # 4-1. 사용자 메시지 표시 및 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4-2. 챗봇 답변 생성 (RAG 모델 호출)
    response = get_rag_answer(prompt)
    
    # 4-3. 챗봇 답변 표시 및 저장
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})