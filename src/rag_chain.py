import os
import re
import logging
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger(__name__)

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

# System prompt
SYSTEM_PROMPT = (
    "You are an expert assistant for the MS in Applied Data Science (MS-ADS) program "
    "at the University of Chicago. Your job is to help prospective students, current "
    "students, and alumni with accurate, helpful answers.\n\n"
    "RULES:\n"
    "1. Answer using ONLY the provided context. Do not use prior knowledge.\n"
    "2. If the context does not contain enough information to answer, say clearly: "
    "'I don't have that information in my knowledge base. Please visit "
    "https://datascience.uchicago.edu or contact the admissions office directly.'\n"
    "3. Always cite the source URL(s) at the end of your answer.\n"
    "4. Use bullet points for lists. Be concise but complete.\n"
    "5. Never invent course names, deadlines, tuition figures, or faculty names.\n"
    "6. If a question is completely unrelated to the MS-ADS program, politely say so.\n\n"
    "CONTEXT:\n{context}"
)

# Question condensation prompt (for follow-up questions)
CONDENSE_PROMPT = (
    "Given the conversation history below and a follow-up question, rewrite the "
    "follow-up as a complete standalone question that can be understood without "
    "the history. Do not answer — only rewrite.\n\n"
    "Conversation history:\n{chat_history}\n\n"
    "Follow-up question: {question}\n"
    "Standalone question:"
)

# PII patterns
PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                             "[SSN REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL REDACTED]"),
    (re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}|\b\d{10}\b"),                "[PHONE REDACTED]"),
]

OFF_TOPIC_KEYWORDS = [
    "stock price", "weather forecast", "recipe", "movie review",
    "bitcoin", "dating", "song lyrics", "sports score",
]


def redact_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def is_off_topic(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in OFF_TOPIC_KEYWORDS)


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[Chunk {i}]\n"
            f"Section: {meta.get('section', 'Unknown')}\n"
            f"Page: {meta.get('page_title', '')}\n"
            f"Source: {meta.get('source', '')}\n\n"
            f"{doc.page_content}"
        )
    return "\n\n" + ("─" * 60 + "\n\n").join(parts)


def extract_sources(docs) -> list:
    seen, urls = set(), []
    for doc in docs:
        url = doc.metadata.get("source", "")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


class RAGChain:
    """
    Conversational RAG chain with:
    - Question condensation (handles follow-up questions)
    - Multi-turn memory (last 10 turns)
    - Source citation in every answer
    - PII redaction
    - Off-topic detection
    """

    def __init__(self, retriever, llm):
        self.retriever    = retriever
        self.llm          = llm
        self.chat_history = []

        self.condense_chain = (
            PromptTemplate.from_template(CONDENSE_PROMPT)
            | self.llm
            | StrOutputParser()
        )

        self.answer_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])
            | self.llm
            | StrOutputParser()
        )

    def _condense_question(self, question: str) -> str:
        """If there is chat history, rewrite follow-up as standalone question."""
        if not self.chat_history:
            return question
        history_str = "\n".join(
            f"Human: {m.content}" if isinstance(m, HumanMessage)
            else f"Assistant: {m.content}"
            for m in self.chat_history
        )
        try:
            return self.condense_chain.invoke({
                "chat_history": history_str,
                "question": question,
            })
        except Exception:
            return question  # fall back to original on error

    def ask(self, question: str, k: int = RETRIEVAL_K) -> dict:
        """
        Full RAG query. Returns:
            {
              "answer":  str,
              "sources": list[str],
              "docs":    list[Document],
              "standalone_question": str,
            }
        """
        # Guard: off-topic
        if is_off_topic(question):
            return {
                "answer": (
                    "I specialise in the MS in Applied Data Science program at UChicago. "
                    "I'm not able to help with that topic. Please ask about courses, "
                    "admissions, faculty, tuition, or career outcomes."
                ),
                "sources": [], "docs": [], "standalone_question": question,
            }

        # Step 1: Condense follow-up questions
        standalone_q = self._condense_question(question)
        log.debug("Standalone question: %s", standalone_q)

        # Step 2: Retrieve relevant chunks
        docs = self.retriever.invoke(standalone_q)

        # Step 3: Generate answer
        raw_answer = self.answer_chain.invoke({
            "context": format_docs(docs),
            "chat_history": self.chat_history,
            "question": standalone_q,
        })

        # Step 4: Post-process
        answer  = redact_pii(raw_answer)
        sources = extract_sources(docs)

        # Append source footnote if not already present
        if sources and "**Sources:**" not in answer:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

        # Step 5: Update memory (keep last 10 turns = 20 messages)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return {
            "answer": answer,
            "sources": sources,
            "docs": docs,
            "standalone_question": standalone_q,
        }

    def reset(self) -> None:
        """Clear conversation history."""
        self.chat_history = []
        log.info("Conversation history cleared.")


def build_rag_chain(vsm, llm) -> RAGChain:
    return RAGChain(retriever=vsm.retriever(k=RETRIEVAL_K), llm=llm)
