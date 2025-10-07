from pymilvus import connections, Collection
import streamlit as st

def get_collection(collection_name: str):
    connections.connect(
        alias="default",
        uri=st.secrets["ZILLIZ_URI"],
        user=st.secrets["ZILLIZ_USER"],
        password=st.secrets["ZILLIZ_PASSWORD"],
        token=st.secrets["ZILLIZ_TOKEN"]
    )
    return Collection(collection_name)
