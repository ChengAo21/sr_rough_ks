"""Package the complete, portable GitHub source tree and verify its contents."""
from pathlib import Path
import hashlib
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"release/JFE_rough_codes_streamlit_ready.zip"
EXCLUDED={".runtime", ".venv", "__pycache__", ".git", "release", ".pytest_cache"}
TEXT_SUFFIXES={".md", ".py", ".cjs", ".js", ".json", ".ipynb", ".csv", ".txt", ".ps1", ".toml", ".svg", ".cff", ".bib"}
HAN=re.compile("[\u3400-\u4dbf\u4e00-\u9fff]")


def main():
    files=sorted(p for p in ROOT.rglob("*") if p.is_file()
        and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
        and p.suffix.lower() not in {".pdf", ".pyc", ".log", ".zip"}
        and p.name not in {"secrets.toml", ".DS_Store"})
    for file in files:
        if HAN.search(file.relative_to(ROOT).as_posix()):
            raise ValueError(f"Non-English filename: {file.relative_to(ROOT)}")
        if file.suffix.lower() in TEXT_SUFFIXES and HAN.search(file.read_text(encoding="utf-8-sig")):
            raise ValueError(f"Untranslated text: {file.relative_to(ROOT)}")
    required=["README.md", "streamlit_app.py", ".gitignore", ".streamlit/config.toml", "LICENSE",
              "requirements.txt", "requirements-pysr.txt", "start_streamlit.ps1", "CITATION.bib",
              "CITATION.cff", "docs/LOCAL_DEMO.md", "docs/STREAMLIT_DEPLOYMENT.md", "code_files/pce_core.py", "results/SR_train.py",
              "results/csvFiles/rough_surface_statistics.csv", "roughlab/sr_worker.py"]
    names={p.relative_to(ROOT).as_posix() for p in files}
    missing=set(required)-names
    if missing: raise ValueError(f"Missing required release files: {sorted(missing)}")
    references=re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",(ROOT/"README.md").read_text(encoding="utf-8"))
    missing_links=[ref for ref in references if not ref.startswith(("http", "#")) and ref.split("#",1)[0] not in names]
    if missing_links: raise ValueError(f"Broken README references: {missing_links}")
    manifest="\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}" for p in files)+"\n"
    ARCHIVE.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE,"w",zipfile.ZIP_DEFLATED) as archive:
        for file in files: archive.write(file,Path(ROOT.name)/file.relative_to(ROOT))
        archive.writestr(f"{ROOT.name}/MANIFEST.sha256",manifest)
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None: raise ValueError("Archive CRC verification failed")
        for line in manifest.splitlines():
            expected,name=line.split("  ",1)
            actual=hashlib.sha256(archive.read(f"{ROOT.name}/{name}")).hexdigest()
            if actual!=expected: raise ValueError(f"Archive checksum mismatch: {name}")
    digest=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    ARCHIVE.with_suffix(".zip.sha256").write_text(f"{digest}  {ARCHIVE.name}\n",encoding="ascii")
    print(f"Archive: {ARCHIVE}")
    print(f"Files: {len(files)+1}; size: {ARCHIVE.stat().st_size/1024**2:.2f} MiB")
    print(f"SHA-256: {digest}")
    print("English text scan, required-file checks, CRC and all file checksums passed.")


if __name__=="__main__": main()
