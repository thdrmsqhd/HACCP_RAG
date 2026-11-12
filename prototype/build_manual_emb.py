"""매뉴얼 페이지 임베딩 생성. uv run --with fastembed python build_manual_emb.py"""
import json
import pathlib

from manual_index import load_pages

OUT = pathlib.Path(__file__).with_name("manual_embeddings.json")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    from fastembed import TextEmbedding

    pages = load_pages()
    texts = [p["text"][:2000] for p in pages]
    model = TextEmbedding(model_name=MODEL)
    vecs = [list(map(float, v)) for v in model.embed(texts)]
    payload = {"model": MODEL, "items": [
        {"page": p["page"], "vec": v} for p, v in zip(pages, vecs)]}
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"saved {OUT} pages={len(vecs)} dims={len(vecs[0])}")


if __name__ == "__main__":
    main()
