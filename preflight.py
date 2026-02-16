"""Pre-flight check for the RAG system."""
import os
import sys

print("=" * 55)
print("      RAG SYSTEM PRE-FLIGHT CHECK")
print("=" * 55)

errors = 0

# 1. Model file
model_path = r"D:\RAG\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
if os.path.exists(model_path):
    size_gb = round(os.path.getsize(model_path) / (1024**3), 2)
    print("[OK]   LLaMA model: {} GB".format(size_gb))
else:
    print("[FAIL] LLaMA model NOT FOUND at " + model_path)
    errors += 1

# 2. .env file
if os.path.exists(r"D:\RAG\.env"):
    print("[OK]   .env file exists")
else:
    print("[FAIL] .env file missing")
    errors += 1

# 3. Key packages
pkgs = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "faiss": "faiss",
    "torch": "torch",
    "sentence_transformers": "sentence-transformers",
    "llama_cpp": "llama-cpp-python",
    "bcrypt": "bcrypt",
    "tiktoken": "tiktoken",
    "structlog": "structlog",
    "prometheus_client": "prometheus-client",
    "FlagEmbedding": "FlagEmbedding",
}
for imp_name, pip_name in pkgs.items():
    try:
        __import__(imp_name)
        print("[OK]   " + pip_name)
    except ImportError:
        print("[FAIL] " + pip_name + " — not installed")
        errors += 1

# 4. CUDA
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        print("[OK]   CUDA: {} ({} GB VRAM)".format(name, vram))
    else:
        print("[WARN] No CUDA available — will use CPU (slower)")
except Exception as e:
    print("[WARN] CUDA check failed: " + str(e))

# 5. Config loads
try:
    sys.path.insert(0, r"D:\RAG")
    from config import settings
    jwt_ok = settings.JWT_SECRET != "change-this-to-a-strong-random-secret"
    print("[OK]   Config loads OK")
    print("[OK]   JWT secret configured: " + str(jwt_ok))
    print("[OK]   Model path: " + settings.LLAMA_MODEL_PATH)
except Exception as e:
    print("[FAIL] Config error: " + str(e))
    errors += 1

# 6. Data dirs
for d in ["data", "data/documents"]:
    path = os.path.join(r"D:\RAG", d)
    if os.path.isdir(path):
        print("[OK]   " + d + "/")
    else:
        print("[WARN] " + d + "/ missing (will be created)")

# 7. Sample docs
doc_dir = os.path.join(r"D:\RAG", "data", "documents")
if os.path.isdir(doc_dir):
    docs = os.listdir(doc_dir)
    print("[OK]   {} sample document(s) found".format(len(docs)))

print()
print("=" * 55)
if errors == 0:
    print("  ALL CHECKS PASSED — READY TO START!")
else:
    print("  {} CHECK(S) FAILED — FIX BEFORE STARTING".format(errors))
print("=" * 55)
