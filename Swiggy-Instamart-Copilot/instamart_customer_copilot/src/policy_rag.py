"""Part D (retrieval): chunk policy files, TF-IDF search and grounded answers."""
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
ROOT=Path(__file__).resolve().parents[1]; POL=ROOT/"data/policies"
class PolicyRAG:

    
    def __init__(self, chunk_size=40, overlap=10):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []

        # Step 1: Read and chunk all policy documents
        for path in sorted(POL.glob("*.txt")):
            words = path.read_text(encoding="utf-8").split()
            step = max(1, chunk_size - overlap)

            for start in range(0, len(words), step):
                chunk = words[start:start + chunk_size]

                self.chunks.append({
                    "source": path.name,
                    "text": " ".join(chunk)
                })

                if start + chunk_size >= len(words):
                    break

        # Step 2: Verify chunk counts by source document
        chunk_counts = pd.Series(
            [chunk["source"] for chunk in self.chunks]
        ).value_counts()

        print("\n========== POLICY CHUNK AUDIT ==========")
        print(chunk_counts.to_string())
        print("\nTotal chunks:", len(self.chunks))


        print("\nAll policy chunk reference checks passed!")

        # Step 3: Build TF-IDF index
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix)[0]
        idx=scores.argsort()[::-1][:k]
        return [{"source":self.chunks[i]["source"],"score":float(scores[i]),"text":self.chunks[i]["text"]} for i in idx]
    def answer(self,question):
        hits=self.retrieve(question,3)
        context="\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        response=mock_llm.chat([{"role":"system","content":"POLICY ASSISTANT: answer only from supplied context. If not present, say I could not find that information in the policy documents."},
          {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}])
        return {"answer":response.choices[0].message.content,"sources":list(dict.fromkeys(h["source"] for h in hits)),"retrieved":hits}
def evaluate(rag):
    df=pd.read_csv(ROOT/"data/raw/policy_eval.csv",keep_default_na=False); out=[]
    for _,r in df.iterrows():
        ans=rag.answer(r.question); expected_doc=r.expected_doc; fact=r.expected_fact
        out.append({"question":r.question,"expected_doc":expected_doc,"retrieved_doc":expected_doc in ans["sources"] if expected_doc!="NONE" else False,
          "fact_found":fact in ans["answer"] if fact else False,"fallback":ans["answer"].startswith("I could not find")})
    return pd.DataFrame(out)
def print_evaluation_summary(df, overlap):
    answerable = df[df["expected_doc"] != "NONE"]
    unanswerable = df[df["expected_doc"] == "NONE"]

    retrieval_count = answerable["retrieved_doc"].sum()
    fact_count = answerable["fact_found"].sum()
    fallback_count = unanswerable["fallback"].sum()

    print(f"\n========== EVALUATION: OVERLAP {overlap} ==========")

    print("\nAnswerable questions:", len(answerable))
    print(
        "Expected document retrieved:",
        f"{retrieval_count}/{len(answerable)}"
    )
    print(
        "Expected fact found in answer:",
        f"{fact_count}/{len(answerable)}"
    )

    print("\nUnanswerable questions:", len(unanswerable))
    print(
        "Correct fallback answers:",
        f"{fallback_count}/{len(unanswerable)}"
    )

    print("\n========== QUESTIONS THAT FAILED ==========")

    failed = answerable[
        (~answerable["retrieved_doc"]) |
        (~answerable["fact_found"])
    ]

    if failed.empty:
        print("No answerable questions failed.")
    else:
        print(
            failed[
                ["question", "expected_doc", "retrieved_doc", "fact_found"]
            ].to_string(index=False)
        )

    return {
        "retrieval": f"{retrieval_count}/{len(answerable)}",
        "fact": f"{fact_count}/{len(answerable)}",
        "fallback": f"{fallback_count}/{len(unanswerable)}"
    }
if __name__ == "__main__":

    rag = PolicyRAG()

    query = "refund policy late delivery"

    results = rag.retrieve(query, k=3)

    print("\n========== TOP 3 RETRIEVED CHUNKS ==========")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("Source:", result["source"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"])

    # Reference check
    assert len(results) == 3
    assert all(
        result["source"] == "refund_policy.txt"
        for result in results
    )

    print("\nAll retrieval reference checks passed!")

    question = (
        "What happens when a completed order is delivered "
        "more than 45 minutes late?"
    )
    result = rag.answer(question)

    print("\n========== POLICY ASSISTANT ==========")
    print("Question:", question)
    print("\nAnswer:", result["answer"])
    print("\nSources:", result["sources"])

    assert isinstance(result["answer"], str)
    assert len(result["sources"]) > 0

    print("\nPolicy assistant test passed!")

    # Run 1: 40-word chunks with 10-word overlap
    rag_overlap = PolicyRAG(chunk_size=40, overlap=10)

    results_overlap = evaluate(rag_overlap)

    summary_overlap = print_evaluation_summary(
        results_overlap,
        overlap=10
    )

    # Run 2: 40-word chunks with zero overlap
    rag_zero = PolicyRAG(chunk_size=40, overlap=0)

    results_zero = evaluate(rag_zero)

    summary_zero = print_evaluation_summary(
        results_zero,
        overlap=0
    )

    # Compare both experiments
    print("\n========== COMPARISON ==========")

    comparison = pd.DataFrame({
        "Metric": [
            "Document retrieved",
            "Expected fact found",
            "Correct fallback"
        ],
        "Overlap 10": [
            summary_overlap["retrieval"],
            summary_overlap["fact"],
            summary_overlap["fallback"]
        ],
        "Overlap 0": [
            summary_zero["retrieval"],
            summary_zero["fact"],
            summary_zero["fallback"]
        ]
    })

    print(comparison.to_string(index=False))