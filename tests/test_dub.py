import json
import subprocess

from scripts.dub import (
    _get_git_state,
    _is_bad_synthesis,
    build_stage_report,
    pick_best_synthesis,
    save_transcript_and_translation,
    update_dashboard,
    words_from_whisperx_transcript,
)


def _init_scratch_git_repo(repo_dir) -> None:
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    # autoria fixa — repo descartável de teste, não precisa da identidade
    # real do usuário, e CI não tem `user.name`/`user.email` configurados.
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)


def test_words_from_whisperx_transcript_flattens_and_orders_by_time() -> None:
    transcript = {
        "segments": [
            {
                "words": [
                    {"word": "world", "start": 1.0, "end": 1.5, "speaker": "SPEAKER_00"},
                ]
            },
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                ]
            },
        ]
    }
    words = words_from_whisperx_transcript(transcript)
    assert [w.text for w in words] == ["Hello", "world"]
    assert words[0].start == 0.0


def test_words_from_whisperx_transcript_skips_words_without_timestamps() -> None:
    transcript = {
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                    {"word": "uh"},
                ]
            }
        ]
    }
    words = words_from_whisperx_transcript(transcript)
    assert [w.text for w in words] == ["Hello"]


def test_words_from_whisperx_transcript_empty_segments_returns_empty_list() -> None:
    assert words_from_whisperx_transcript({"segments": []}) == []


def test_build_stage_report_empty_segments() -> None:
    assert build_stage_report([]) == {"total_segments": 0}


def test_build_stage_report_computes_rates() -> None:
    results = [
        {
            "translated_ok": True,
            "synthesized_ok": True,
            "within_duration_tolerance": True,
            "evaluated_ok": True,
            "any_flag": False,
            "synthesis_attempts": 1,
        },
        {
            "translated_ok": True,
            "synthesized_ok": False,
            "within_duration_tolerance": False,
            "evaluated_ok": False,
            "any_flag": True,
            "synthesis_attempts": 2,
        },
    ]
    report = build_stage_report(results)
    assert report["total_segments"] == 2
    assert report["translated_rate"] == 1.0
    assert report["synthesized_rate"] == 0.5
    assert report["segments_needing_synth_retry_rate"] == 0.5
    assert report["within_duration_tolerance_rate"] == 0.5
    assert report["evaluated_rate"] == 0.5
    assert report["segments_with_any_flag_rate"] == 0.5


def test_is_bad_synthesis_true_for_repetition() -> None:
    reference = "Já dei o troco pra muitos na vida."
    hypothesis = "Já dei o troco pra muitos na vida. Já dei o troco pra muitos na vida."
    assert _is_bad_synthesis(reference, hypothesis) is True


def test_is_bad_synthesis_true_for_truncation() -> None:
    reference = "Tudo que eles deram pra isso tá valendo zero"
    hypothesis = "Tudo que eles deram pra isso"
    assert _is_bad_synthesis(reference, hypothesis) is True


def test_is_bad_synthesis_true_for_intermediate_cer_without_repetition_or_truncation() -> None:
    # T0.14: o predicado antigo (CER >= 1.0) não disparava aqui — só pegava
    # falha catastrófica. Agora usa cer_alto (limiar do T0.13).
    reference = "Tudo que eles deram pra isso ta valendo zero"
    hypothesis = "Tua que ele deu pra isso ta valeu zero"
    assert _is_bad_synthesis(reference, hypothesis) is True


def test_is_bad_synthesis_false_for_clean_match() -> None:
    reference = "Eles tinham força pra cima de todos nós juntos."
    hypothesis = "Eles tinham força pra cima de todos nós juntos."
    assert _is_bad_synthesis(reference, hypothesis) is False


def test_pick_best_synthesis_picks_first_clean_candidate() -> None:
    reference = "se não ficarmos juntos"
    hypotheses = [
        "Se não ficarmos juntos, se não ficarmo juntos.",  # repetido
        "Se não ficarmos juntos,",  # limpo
        "Se não ficarmos juntos, juntos,",  # repetido
    ]
    assert pick_best_synthesis(reference, hypotheses) == 1


def test_pick_best_synthesis_falls_back_to_lowest_cer_when_all_bad() -> None:
    reference = "Eles tinham força pra cima de todos nós juntos."
    hypotheses = [
        "",  # CER 1.0, o pior
        "Eles tinham força pra cima de todos nós juntos e mais um pouco de coisa aleatória.",
    ]
    # o segundo tem CER > 0 mas bem menor que o primeiro (vazio == CER 1.0)
    assert pick_best_synthesis(reference, hypotheses) == 1


def test_pick_best_synthesis_empty_hypotheses_returns_zero() -> None:
    assert pick_best_synthesis("qualquer coisa", []) == 0


def test_pick_best_synthesis_single_candidate() -> None:
    assert pick_best_synthesis("oi", ["oi"]) == 0


def test_pick_best_synthesis_prefers_on_duration_over_clean_but_wildly_long() -> None:
    # Achado real (beast.mp4, seg0009): cand0 tinha texto limpo o bastante
    # pra passar em _is_bad_synthesis mas saiu 3x mais longo que o alvo
    # (cauda alucinada); cand1 bateu a duração quase exata. Sem dado de
    # duração, a versão antiga escolhia cand0 (primeiro "limpo") e estourava
    # a timeline nos segmentos seguintes — pior que qualquer flag de
    # conteúdo.
    reference = "Tchau, galera! Vamos dormir."
    hypotheses = [
        "Tchau, galera. Vamos la. Tchau gaia. Tchau gays. Tchau gangues. Tchau guys.",
        "Tchau, galera! Vamos dormir.",
    ]
    winner = pick_best_synthesis(
        reference,
        hypotheses,
        achieved_seconds=[31.52, 10.72],
        target_seconds=10.27,
        tolerance=0.08,
    )
    assert winner == 1


def test_pick_best_synthesis_prefers_on_duration_even_with_suspect_content() -> None:
    # Nenhum candidato passa em _is_bad_synthesis, mas um bate a duração —
    # isso ainda vence, porque estourar a timeline é pior que uma flag de
    # qualidade de conteúdo (ver docstring de pick_best_synthesis).
    reference = "Eles tinham força pra cima de todos nós juntos."
    hypotheses = [
        "Eles tinham forca pra cima de todos nos juntos e mais um pouco de coisa aleatoria a mais.",
        "Eles tinham forca pra cima de todos nos juntos e mais um pouco de coisa aleatoria a mais.",
    ]
    winner = pick_best_synthesis(
        reference,
        hypotheses,
        achieved_seconds=[20.0, 3.0],
        target_seconds=3.0,
        tolerance=0.08,
    )
    assert winner == 1


def test_pick_best_synthesis_without_duration_data_ignores_duration() -> None:
    # Compatibilidade: sem achieved_seconds/target_seconds, comportamento
    # idêntico ao antigo (só conteúdo) — quem chama sem esse dado não muda.
    reference = "se não ficarmos juntos"
    hypotheses = [
        "Se não ficarmos juntos, se não ficarmo juntos.",  # repetido
        "Se não ficarmos juntos,",  # limpo
    ]
    assert pick_best_synthesis(reference, hypotheses) == 1


def test_save_transcript_and_translation_writes_both_files(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.5, "text": " hello ", "words": []}]}),
        encoding="utf-8",
    )
    report = {
        "segments": [
            {
                "id": "seg0000",
                "t_inicio": 0.0,
                "t_fim": 1.5,
                "texto_original": "Hello world",
                "traducao": "Olá mundo",
            }
        ]
    }
    out_path = tmp_path / "video.dub.mp4"

    transcript_path, translation_path = save_transcript_and_translation(report, cache_dir, out_path)

    assert transcript_path.name == "video.dub_transcript.json"
    # simplificado (T0.12b: sem detalhe por palavra, indentado) e sem
    # espaços de sobra no texto (WhisperX costuma prefixar com um espaço)
    assert json.loads(transcript_path.read_text(encoding="utf-8")) == [
        {"start": 0.0, "end": 1.5, "text": "hello"}
    ]
    assert "\n" in transcript_path.read_text(encoding="utf-8")  # indentado, não numa linha só
    assert translation_path.name == "video.dub_traducao.json"
    rows = json.loads(translation_path.read_text(encoding="utf-8"))
    assert rows == [
        {
            "id": "seg0000",
            "t_inicio": 0.0,
            "t_fim": 1.5,
            "texto_original": "Hello world",
            "traducao_usada": "Olá mundo",
        }
    ]


def test_save_transcript_and_translation_skips_missing_transcript(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    report = {"segments": []}
    out_path = tmp_path / "video.dub.mp4"

    transcript_path, _ = save_transcript_and_translation(report, cache_dir, out_path)

    assert not transcript_path.exists()


def test_get_git_state_clean_tree_reports_not_dirty(tmp_path) -> None:
    repo_dir = tmp_path / "repo"
    _init_scratch_git_repo(repo_dir)
    (repo_dir / "file.txt").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo_dir, check=True)
    expected_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()

    sha, dirty = _get_git_state(repo_dir)

    assert sha == expected_sha
    assert dirty is False


def test_get_git_state_modified_file_reports_dirty(tmp_path) -> None:
    repo_dir = tmp_path / "repo"
    _init_scratch_git_repo(repo_dir)
    (repo_dir / "file.txt").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo_dir, check=True)

    (repo_dir / "file.txt").write_text("v2 sem commitar", encoding="utf-8")
    _, dirty = _get_git_state(repo_dir)

    assert dirty is True


def test_get_git_state_non_git_directory_returns_none_without_raising(tmp_path) -> None:
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()

    sha, dirty = _get_git_state(not_a_repo)

    assert sha is None
    assert dirty is False


def test_update_dashboard_history_entry_includes_git_sha_and_dirty(tmp_path) -> None:
    repo_dir = tmp_path / "repo"
    _init_scratch_git_repo(repo_dir)
    (repo_dir / "file.txt").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo_dir, check=True)
    expected_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()

    report = {"video": "video.mp4", "output": "video.dub.mp4", "segments": []}
    reports_dir = tmp_path / "reports"

    update_dashboard(report, reports_dir, repo_dir=repo_dir)
    history = json.loads((reports_dir / "history.json").read_text(encoding="utf-8"))
    assert history[0]["git_sha"] == expected_sha
    assert history[0]["git_dirty"] is False

    # segunda execução com a working tree sujo — cada entrada reflete o
    # estado do git NO MOMENTO daquela execução, não é recalculada depois
    (repo_dir / "file.txt").write_text("v2 sem commitar", encoding="utf-8")
    update_dashboard(report, reports_dir, repo_dir=repo_dir)
    history = json.loads((reports_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert history[0]["git_dirty"] is False
    assert history[1]["git_dirty"] is True
