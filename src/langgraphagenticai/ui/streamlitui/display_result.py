import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def _clean_response(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        if usecase == "Basic Chatbot":
            with st.chat_message("user"):
                st.write(user_message)

            for event in graph.stream({
                "messages": [HumanMessage(content=user_message)]
            }):
                for value in event.values():
                    messages = value.get("messages")

                    if isinstance(messages, list):
                        last_message = messages[-1]
                    else:
                        last_message = messages

                    if isinstance(last_message, AIMessage):
                        cleaned_output = self._clean_response(last_message.content)
                        with st.chat_message("assistant"):
                            st.write(cleaned_output)

        elif usecase == "Chatbot With Web":
            with st.chat_message("user"):
                st.write(user_message)

            initial_state = {
                "messages": [HumanMessage(content=user_message)]
            }

            res = graph.invoke(initial_state)

            for message in res["messages"]:

                if isinstance(message, ToolMessage):
                    with st.chat_message("assistant"):
                        st.write("🔧 Tool Call Start")
                        st.write(message.content)
                        st.write("🔧 Tool Call End")

                elif isinstance(message, AIMessage) and message.content:
                    cleaned_output = self._clean_response(message.content)
                    with st.chat_message("assistant"):
                        st.write(cleaned_output)