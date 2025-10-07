from langchain_core.prompts import ChatPromptTemplate
from langchain_fireworks import ChatFireworks

def classify_intent(user_input: str, fireworks_api_key: str) -> str:
    llm = ChatFireworks(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        fireworks_api_key=fireworks_api_key,
        temperature=0
    )

    system_prompt = """
    You are an intent classifier for BPS chatbot.
    Categories:
    - publications: publikasi, laporan, dokumen
    - press: press release (berita resmi statistik BPS)
    - news: news, berita acara, kegiatan
    - general: profil, layanan, kebijakan, umum
    - tables: tabel statistik (statistik tabel BPS seperti table pages)
    - other: pertanyaan di luar cakupan BPS data atau tidak relevan
    Output: one word only. Choose the best fitting category.
    If nothing matches, output "other".
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"question": user_input})
    return response.content.strip().lower()
