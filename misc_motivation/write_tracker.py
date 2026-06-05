import subprocess
import datetime as dt
import csv
import re
from pathlib import Path

# =========================
# CONFIG
# =========================
TEX_MAIN = "main.tex"

LOG_FILE = Path("word_history.csv")
CHAPTER_LOG = Path("chapter_history.csv")
GIT_LOG = Path("git_words.csv")


# =========================
# WORD COUNT (ROBUST TEXCOUNT PARSER)
# =========================
def get_word_count(tex_file=TEX_MAIN) -> int:
    cmd = ["texcount", "-inc", "-sum", "-1", tex_file]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    output = result.stdout.strip()

    # Extract FIRST integer only (handles: "12102 (errors:4)")
    match = re.search(r"\d+", output)
    if not match:
        raise ValueError(f"Unexpected texcount output: {output}")

    return int(match.group())


# =========================
# CHAPTER / SECTION TRACKING
# =========================
def get_chapter_breakdown(tex_file=TEX_MAIN):
    """
    Lightweight structural scan of LaTeX file.
    Works with \include, \chapter, \section
    """

    pattern = re.compile(r"\\(chapter|section|subsection)\{(.+?)\}")

    counts = {}

    with open(tex_file, "r", encoding="utf-8") as f:
        text = f.read()

    for match in pattern.finditer(text):
        label = f"{match.group(1)}:{match.group(2)}"
        counts[label] = counts.get(label, 0) + 1

    return counts


# =========================
# SAFE GIT ANALYSIS (NO CHECKOUT)
# =========================
def get_git_commits(limit=50):
    cmd = ["git", "log", f"-{limit}", "--pretty=format:%H|%ad", "--date=short"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    commits = []
    for line in out.split("\n"):
        if "|" in line:
            h, d = line.split("|", 1)
            commits.append((h, d))

    return commits


def get_words_at_commit(commit_hash):
    """
    SAFE METHOD: uses git archive instead of checkout
    """

    cmd = ["git", "archive", commit_hash, "main.tex"]
    proc = subprocess.run(cmd, capture_output=True)

    if proc.returncode != 0:
        return None

    # write temp file
    import tarfile
    import io

    tar_bytes = io.BytesIO(proc.stdout)

    try:
        with tarfile.open(fileobj=tar_bytes) as tar:
            member = tar.extractfile("main.tex")
            if not member:
                return None

            content = member.read().decode("utf-8")

        # write temp and count
        tmp = Path(".__tmp.tex")
        tmp.write_text(content, encoding="utf-8")

        return get_word_count(str(tmp))

    except:
        return None


def log_git_history():
    commits = get_git_commits()

    with open(GIT_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["commit", "date", "words"])

        for commit, date in commits:
            words = get_words_at_commit(commit)
            if words is not None:
                writer.writerow([commit, date, words])

    # cleanup
    tmp = Path(".__tmp.tex")
    if tmp.exists():
        tmp.unlink()


# =========================
# DAILY LOG
# =========================
def append_daily():
    today = dt.date.today().isoformat()
    words = get_word_count()

    new_file = not LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow(["date", "words"])

        writer.writerow([today, words])


# =========================
# CHAPTER LOG
# =========================
def log_chapters():
    today = dt.date.today().isoformat()
    breakdown = get_chapter_breakdown()

    new_file = not CHAPTER_LOG.exists()

    with open(CHAPTER_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow(["date", "chapter", "count"])

        for chapter, count in breakdown.items():
            writer.writerow([today, chapter, count])


# =========================
# MASTER RUNNER
# =========================
def run_all():
    append_daily()
    log_chapters()
    # log_git_history()

    print("✔ LaTeX tracking complete:")
    print("  - daily word count logged")
    print("  - chapter structure logged")
    # print("  - git history updated")


if __name__ == "__main__":
    run_all()
