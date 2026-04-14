import os
import io
import base64
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# Data
# =========================
PRODUCT_DATABASE = [
    "Light blue shirt, cotton fabric, suitable for summer wear, sizes from S to XL",
    "Red dress, cotton fabric, suitable for summer wear, sizes from S to XL",
    "White T-shirt, 100% cotton, crew neck short sleeves, versatile style",
    "Black leather shoes, genuine leather, business formal, durable and slip-resistant",
    "Sneakers, lightweight and breathable, suitable for running and fitness, multiple colors available",
]


# =========================
# Config
# =========================

def get_openai_api_key() -> str:
    """Load OpenAI API key from environment."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")
    return api_key


# =========================
# Image Processing
# =========================

def pil_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL image to JPEG bytes."""
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# =========================
# Vision → Text
# =========================

def generate_image_description(
    client: OpenAI,
    image_bytes: bytes,
    model: str = "gpt-4o-mini",
) -> str:
    """Convert image → structured product description."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are an e-commerce product tagger.\n"
        "Describe the product concisely for retrieval.\n"
        "Return 5–10 bullet points covering:\n"
        "- category\n"
        "- color\n"
        "- material\n"
        "- features\n"
        "- use cases\n"
        "Avoid brand names unless visible."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return response.choices[0].message.content.strip()


# =========================
# Vector Store
# =========================

@st.cache_resource
def build_vector_store(api_key: str, model: str) -> FAISS:
    """Create FAISS vector index."""
    docs = [
        Document(page_content=text, metadata={"id": i})
        for i, text in enumerate(PRODUCT_DATABASE)
    ]

    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    return FAISS.from_documents(docs, embeddings)


# =========================
# Retrieval
# =========================

def retrieve_products(
    query: str,
    vectorstore: FAISS,
    top_k: int,
) -> List[Tuple[str, float]]:
    """Retrieve top-k similar products."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [(doc.page_content, score) for doc, score in results]


def display_product_database() -> None:
    """Display the built-in product database."""
    with st.expander("Product database"):
        for i, text in enumerate(PRODUCT_DATABASE, 1):
            st.write(f"{i}. {text}")


# =========================
# UI
# =========================

def main() -> None:
    st.set_page_config(page_title="Product Retrieval", layout="centered")
    st.title("🛍️ Multimodal Product Retrieval")

    try:
        api_key = get_openai_api_key()
    except Exception as e:
        st.error(str(e))
        st.stop()

    client = OpenAI(api_key=api_key)

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top-K", 1, 10, 3)
    vision_model = st.sidebar.selectbox("Vision Model", ["gpt-4o-mini", "gpt-4o"])
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["text-embedding-3-large", "text-embedding-3-small"],
    )

    vectorstore = build_vector_store(api_key, embedding_model)

    uploaded_file = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])
    text_query = st.text_input("Or enter a text query")
    search_clicked = st.button("🔍 Search")

    if not uploaded_file and not text_query:
        st.info("Upload an image OR type a query.")
        display_product_database()
        return

    query: Optional[str] = text_query.strip() if text_query else None

    if search_clicked and uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=300)

        with st.spinner("Analyzing image..."):
            description = generate_image_description(
                client,
                pil_to_bytes(image),
                model=vision_model,
            )

        st.subheader("Generated Description")
        st.write(description)

        if query:
            query = f"{description}\n\nUser query: {query}"
        else:
            query = description

    if search_clicked and query:
        with st.spinner("Retrieving products..."):
            results = retrieve_products(query, vectorstore, top_k)

        st.subheader("Results")
        st.markdown(f"**Search query:** {query}")
        st.caption(f"Returned {len(results)} products.")

        for i, (text, score) in enumerate(results, 1):
            st.markdown(f"**{i}.** {text}")
            st.caption(f"Distance: {score:.4f}")

    if not search_clicked:
        st.info("Press Search to start retrieval.")
        display_product_database()


if __name__ == "__main__":
    main()
