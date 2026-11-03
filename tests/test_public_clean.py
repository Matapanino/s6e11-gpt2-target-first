"""This code goes public later, so no file may carry a machine name, a home path or a private address."""
import re

from conftest import CODE

# Spelled in pieces so that this test does not match itself.
FORBIDDEN = [r"192\.168\.", "develop" + r"er@", "dir_" + "kawamata", "masaya" + "kawamata", "/ho" + r"me/", "/Us" + r"ers/",
             "RT" + r"X\s*3090", "/tm" + "p/claude", "datagrid" + r"\.co"]


def private_files():
    """Files named in .gitignore never go public, so they may hold machine details, such as the state of a live run."""
    return set((CODE / ".gitignore").read_text().split())


def test_the_private_run_notes_are_kept_out_of_git():
    assert "RUN_STATE.md" in private_files()


def test_no_machine_specific_text_in_the_code_folder():
    pattern = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
    problems = []
    for path in sorted(CODE.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".toml", ".sh"} \
                and path.name not in private_files():
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(re.sub(r"https://(?:www\.)?(?:kaggle\.com|github\.com)/[^\s)]+", "", line)):
                    problems.append(f"{path.relative_to(CODE)}:{number}: {line.strip()}")
    assert not problems, "\n".join(problems)
