import os
import pytest

from src.rag import ingest_pdf_into_qdrant, rag_answer


@pytest.mark.integration
def test_rag_flow(tmp_path):
    # Create a tiny PDF-like content by writing text and using pypdf is heavy; assume a prepared sample exists.
    sample_pdf = os.path.join("data", "sample.pdf")
    if not os.path.exists(sample_pdf):
        pytest.skip("sample.pdf not available in data/")

    ingest_pdf_into_qdrant(sample_pdf)
    res = rag_answer("What is the document about?")
    assert isinstance(res["answer"], str) and len(res["answer"]) > 0


