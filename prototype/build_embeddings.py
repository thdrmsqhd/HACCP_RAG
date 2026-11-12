"""조문 임베딩 생성 (fastembed). 실행: uv run --with fastembed python build_embeddings.py"""
import json
import pathlib

from corpus import load_articles

GOSI = r"C:\Users\gbsong\Documents\HACCP_RAG\corpus\01_공개자료\식품및축산물안전관리인증기준_고시_2026-60호.txt"
OUT = pathlib.Path(__file__).with_name("embeddings.json")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    from fastembed import TextEmbedding

    arts = load_articles(GOSI)
    texts = [f"제{a['num']}조 {a['title']}. {a['text']}" for a in arts]
    model = TextEmbedding(model_name=MODEL)
    vecs = [list(map(float, v)) for v in model.embed(texts)]

    payload = {
        "model": MODEL,
        "dims": len(vecs[0]),
        "items": [{"num": a["num"], "title": a["title"], "vec": v} for a, v in zip(arts, vecs)],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"saved {OUT} articles={len(vecs)} dims={len(vecs[0])}")


if __name__ == "__main__":
    main()
