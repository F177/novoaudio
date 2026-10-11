from scripts.check_generalization import find_leaks, ngrams, parse_diff, threshold_changes, words

DIFF = """\
+++ b/packages/pipeline/translation.py
-MAX_ALTERNATIVES = 8
+MAX_ALTERNATIVES = 12
-_BUDGET_WEIGHT = 0.6
+_BUDGET_WEIGHT = 0.9
+FIXES = {"does your mom cook for you": "sua mãe cozinha pra você"}
+    if video == "clip-ficticio":
+    return f"seg{i:04d}"
+++ b/tests/test_translation.py
+    assert fix("does your mom cook for you") == "x"
+++ b/packages/pipeline/quality.py
+    special = "seg0006"
"""

CORPUS = ngrams(words("Listen. Does your mom cook for you? Every single night?"))


def test_text_name_and_segment_id_leaks() -> None:
    added, _ = parse_diff(DIFF)
    found = find_leaks(added, CORPUS, {"clip-ficticio"})
    by_path = {(f.path, f.severity) for f in found}
    assert ("packages/pipeline/translation.py", "bloqueia") in by_path
    assert ("tests/test_translation.py", "aviso") in by_path  # teste: só aviso
    assert any("seg0006" in f.detail for f in found)
    assert not any("seg{i:04d}" in f.detail for f in found)  # formato, não id
    assert any("clip-ficticio" in f.detail for f in found)


def test_threshold_change_is_a_warning() -> None:
    added, removed = parse_diff(DIFF)
    found = threshold_changes(added, removed)
    assert {f.severity for f in found} == {"aviso"}
    assert any("8 → 12" in f.detail for f in found)
    assert any("_BUDGET_WEIGHT: 0.6 → 0.9" in f.detail for f in found)  # constante privada


def test_short_common_phrases_do_not_trigger() -> None:
    added, _ = parse_diff("+++ b/packages/x.py\n+# you have to pay\n")
    assert find_leaks(added, CORPUS, set()) == []


def test_training_dataset_with_eval_text_is_blocked(tmp_path) -> None:
    from scripts.check_generalization import scan_dataset

    data = tmp_path / "train.jsonl"
    data.write_text('{"text": "Uma frase qualquer de treino."}\n'
                    '{"text": "Listen. Does your mom cook for you? ok"}\n', encoding="utf-8")
    [finding] = scan_dataset([data], CORPUS)
    assert finding.severity == "bloqueia" and finding.path.endswith("train.jsonl:2")


def test_evaluation_data_itself_is_not_scanned() -> None:
    added, _ = parse_diff("+++ b/evals/human_labels/en-xx/clip-ficticio.jsonl\n"
                          '+{"video": "clip-ficticio", "segment_id": "seg0006"}\n')
    assert find_leaks(added, CORPUS, {"clip-ficticio"}) == []
