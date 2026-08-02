import asyncio
import json
import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Local PDF RAG",
    page_icon="📄",
    layout="centered",
)

INNGEST_API_BASE = os.getenv(
    "INNGEST_API_BASE",
    "http://127.0.0.1:8288/v1",
)


# ---------------------------------------------------------
# Inngest helpers
# ---------------------------------------------------------

def create_inngest_client() -> inngest.Inngest:
    """
    Create a new client for each async operation.

    We do not cache this client because Streamlit may close
    and recreate event loops between application reruns.
    """
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
    )


async def send_rag_ingest_event(
    pdf_path: Path,
    source_id: str,
) -> str:
    """Send a PDF ingestion event and return its event ID."""

    client = create_inngest_client()

    event_ids = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": source_id,
            },
        )
    )

    if not event_ids:
        raise RuntimeError(
            "Inngest did not return an event ID."
        )

    return event_ids[0]


async def send_rag_query_event(
    question: str,
    top_k: int,
) -> str:
    """Send a question event and return its event ID."""

    client = create_inngest_client()

    event_ids = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    if not event_ids:
        raise RuntimeError(
            "Inngest did not return an event ID."
        )

    return event_ids[0]


# ---------------------------------------------------------
# PDF file handling
# ---------------------------------------------------------

def save_uploaded_pdf(uploaded_file) -> Path:
    """Save the uploaded PDF inside the local uploads folder."""

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Prevent the filename from containing another path.
    safe_filename = Path(uploaded_file.name).name
    file_path = uploads_dir / safe_filename

    file_path.write_bytes(
        uploaded_file.getvalue()
    )

    return file_path


# ---------------------------------------------------------
# Inngest run polling
# ---------------------------------------------------------

def fetch_runs(event_id: str) -> list[dict]:
    """Get all function runs triggered by an event."""

    url = (
        f"{INNGEST_API_BASE}/events/"
        f"{event_id}/runs"
    )

    response = requests.get(
        url,
        timeout=10,
    )

    response.raise_for_status()

    body = response.json()
    return body.get("data", [])


def parse_output(output) -> dict:
    """
    Convert the Inngest output into a dictionary.

    Depending on the server version, output may already be
    a dictionary or may be returned as a JSON string.
    """

    if isinstance(output, dict):
        return output

    if isinstance(output, str):
        try:
            parsed = json.loads(output)

            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}

    return {}


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 300,
    poll_interval_s: float = 0.75,
) -> dict:
    """
    Wait until the Inngest workflow completes and return
    its final output.
    """

    started_at = time.monotonic()
    last_status = "Waiting for run"

    while time.monotonic() - started_at < timeout_s:
        runs = fetch_runs(event_id)

        if runs:
            # Each event in this project triggers one function.
            run = runs[0]

            status = str(
                run.get("status", "")
            ).strip()

            normalized_status = status.lower()

            if status:
                last_status = status

            if normalized_status in {
                "completed",
                "succeeded",
                "success",
                "finished",
            }:
                return parse_output(
                    run.get("output")
                )

            if normalized_status in {
                "failed",
                "cancelled",
                "canceled",
            }:
                error_details = (
                    run.get("output")
                    or run.get("error")
                    or "No error details were returned."
                )

                raise RuntimeError(
                    f"Inngest workflow {status}: "
                    f"{error_details}"
                )

        time.sleep(poll_interval_s)

    raise TimeoutError(
        "Timed out while waiting for the Inngest workflow. "
        f"Last status: {last_status}"
    )


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "rag_answer" not in st.session_state:
    st.session_state.rag_answer = None

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []


# ---------------------------------------------------------
# PDF ingestion interface
# ---------------------------------------------------------

st.title("📄 Local PDF RAG")
st.write(
    "Upload a PDF, store it in Qdrant, and ask "
    "questions using your local Ollama model."
)

st.subheader("1. Upload and ingest a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

ingest_clicked = st.button(
    "Ingest PDF",
    type="primary",
    disabled=uploaded_file is None,
)

if ingest_clicked and uploaded_file is not None:
    try:
        with st.spinner(
            "Reading, chunking, embedding and storing the PDF..."
        ):
            pdf_path = save_uploaded_pdf(
                uploaded_file
            )

            source_id = Path(
                uploaded_file.name
            ).name

            event_id = asyncio.run(
                send_rag_ingest_event(
                    pdf_path=pdf_path,
                    source_id=source_id,
                )
            )

            output = wait_for_run_output(
                event_id=event_id,
                timeout_s=300,
            )

            ingested_count = output.get(
                "ingested",
                0,
            )

        st.success(
            f"Successfully ingested {ingested_count} "
            f"chunks from {source_id}."
        )

    except requests.RequestException as error:
        st.error(
            "Could not connect to the Inngest development "
            f"server: {error}"
        )

    except Exception as error:
        st.error(
            f"PDF ingestion failed: {error}"
        )


# ---------------------------------------------------------
# Question interface
# ---------------------------------------------------------

st.divider()
st.subheader("2. Ask a question")

with st.form("rag_query_form"):
    question = st.text_area(
        "Your question",
        placeholder=(
            "For example: What are the main ideas "
            "explained in the document?"
        ),
    )

    top_k = st.number_input(
        "Number of document chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    query_submitted = st.form_submit_button(
        "Ask",
        type="primary",
    )


if query_submitted:
    clean_question = question.strip()

    if not clean_question:
        st.warning(
            "Enter a question before pressing Ask."
        )

    else:
        try:
            with st.spinner(
                "Searching Qdrant and generating an answer "
                "with Ollama..."
            ):
                event_id = asyncio.run(
                    send_rag_query_event(
                        question=clean_question,
                        top_k=int(top_k),
                    )
                )

                output = wait_for_run_output(
                    event_id=event_id,
                    timeout_s=300,
                )

                st.session_state.rag_answer = output.get(
                    "answer",
                    ""
                )

                st.session_state.rag_sources = output.get(
                    "sources",
                    []
                )

        except requests.RequestException as error:
            st.error(
                "Could not connect to the Inngest development "
                f"server: {error}"
            )

        except Exception as error:
            st.error(
                f"Question processing failed: {error}"
            )


# ---------------------------------------------------------
# Display the latest answer
# ---------------------------------------------------------

if st.session_state.rag_answer is not None:
    st.subheader("Answer")

    st.write(
        st.session_state.rag_answer
        or "No answer was generated."
    )

    if st.session_state.rag_sources:
        st.caption("Sources")

        for source in st.session_state.rag_sources:
            st.write(f"- {source}")