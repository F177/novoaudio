import json
import subprocess

from scripts.dub import (
    _get_git_state,
    _hash_file,
    _hash_json,
    _is_bad_synthesis,
    _record_stage_hash,
    _stage_needs_rerun,
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


def test_pick_best_synthesis_picks_lowest_cer_among_clean_candidates() -> None:
    # T0.24, achado real: a versão antiga devolvia o PRIMEIRO candidato que
    # passasse em _is_bad_synthesis, mesmo que um candidato POSTERIOR na
    # mesma categoria (limpo + dentro da duração) tivesse CER bem menor.
    # Aqui os dois candidatos são "limpos" (sem repetição/truncamento) e
    # batem a duração, mas cand0 tem erro de transcrição e cand2 é exato —
    # o vencedor deve ser o de menor CER (índice 2), não o primeiro (0).
    reference = "Eles chegaram tarde na festa ontem à noite."
    hypotheses = [
        "Eles chegaram tarde na festa ontem à noita.",  # 1 char errado
        "Eles chegaram tarde na festa ontem à noite e ficaram pouco.",  # mais longo, CER maior
        "Eles chegaram tarde na festa ontem à noite.",  # exato, CER 0
    ]
    winner = pick_best_synthesis(
        reference,
        hypotheses,
        achieved_seconds=[5.0, 5.0, 5.0],
        target_seconds=5.0,
        tolerance=0.08,
    )
    assert winner == 2


def test_pick_best_synthesis_ties_on_cer_break_by_closest_duration() -> None:
    # Dois candidatos com o mesmo texto (mesmo CER) — o desempate é pela
    # duração mais próxima do alvo, não pelo primeiro da lista.
    reference = "Vamos embora agora."
    hypotheses = ["Vamos embora agora.", "Vamos embora agora."]
    winner = pick_best_synthesis(
        reference,
        hypotheses,
        achieved_seconds=[3.5, 3.05],
        target_seconds=3.0,
        tolerance=0.5,
    )
    assert winner == 1


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


def test_hash_json_is_deterministic_regardless_of_key_order() -> None:
    assert _hash_json({"a": 1, "b": 2}) == _hash_json({"b": 2, "a": 1})


def test_hash_json_differs_for_different_content() -> None:
    assert _hash_json({"a": 1}) != _hash_json({"a": 2})


def test_hash_file_differs_for_different_content(tmp_path) -> None:
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"conteudo 1")
    f2.write_bytes(b"conteudo 2 diferente")
    assert _hash_file(f1) != _hash_file(f2)


def test_hash_file_same_content_same_hash(tmp_path) -> None:
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"mesmo conteudo")
    f2.write_bytes(b"mesmo conteudo")
    assert _hash_file(f1) == _hash_file(f2)


def test_stage_needs_rerun_true_when_output_missing(tmp_path) -> None:
    missing = tmp_path / "out.json"
    assert _stage_needs_rerun(
        tmp_path, "translate", "abc", missing, force=False, force_stage=None
    )


def test_stage_needs_rerun_true_when_no_hash_recorded_even_if_output_exists(tmp_path) -> None:
    # Cache de antes da T0.25 (ou primeira rodada): arquivo de saída existe,
    # mas não há prova de que veio da mesma entrada — trata como obsoleto.
    output = tmp_path / "out.json"
    output.write_text("{}", encoding="utf-8")
    assert _stage_needs_rerun(
        tmp_path, "translate", "abc", output, force=False, force_stage=None
    )


def test_stage_needs_rerun_false_when_hash_matches(tmp_path) -> None:
    output = tmp_path / "out.json"
    output.write_text("{}", encoding="utf-8")
    _record_stage_hash(tmp_path, "translate", "abc")
    assert not _stage_needs_rerun(
        tmp_path, "translate", "abc", output, force=False, force_stage=None
    )


def test_stage_needs_rerun_true_when_hash_differs(tmp_path) -> None:
    output = tmp_path / "out.json"
    output.write_text("{}", encoding="utf-8")
    _record_stage_hash(tmp_path, "translate", "abc")
    assert _stage_needs_rerun(
        tmp_path, "translate", "xyz", output, force=False, force_stage=None
    )


def test_stage_needs_rerun_true_when_forced_globally(tmp_path) -> None:
    output = tmp_path / "out.json"
    output.write_text("{}", encoding="utf-8")
    _record_stage_hash(tmp_path, "translate", "abc")
    assert _stage_needs_rerun(
        tmp_path, "translate", "abc", output, force=True, force_stage=None
    )


def test_stage_needs_rerun_force_stage_only_affects_named_stage(tmp_path) -> None:
    # --force-stage synthesize: só a etapa nomeada é forçada, as outras
    # continuam respeitando o hash normalmente.
    synth_output = tmp_path / "synth.json"
    synth_output.write_text("{}", encoding="utf-8")
    _record_stage_hash(tmp_path, "synthesize", "abc")
    assert _stage_needs_rerun(
        tmp_path, "synthesize", "abc", synth_output, force=False, force_stage="synthesize"
    )

    translate_output = tmp_path / "translate.json"
    translate_output.write_text("{}", encoding="utf-8")
    _record_stage_hash(tmp_path, "translate", "def")
    assert not _stage_needs_rerun(
        tmp_path, "translate", "def", translate_output, force=False, force_stage="synthesize"
    )


def test_changing_segmentation_invalidates_translate_and_synth_but_not_upstream_stages(
    tmp_path,
) -> None:
    # T0.25, cenário de aceite literal: trocar a segmentação invalida
    # tradução e síntese (o conteúdo que essas etapas recebem muda), mas
    # NÃO separação de stems nem ASR (a entrada delas é o áudio/vocals, que
    # não mudou — só a segmentação local mudou).
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio fake, nao mudou entre as duas rodadas")
    vocals_path = tmp_path / "vocals.wav"
    vocals_path.write_bytes(b"vocals fake, tambem nao mudou")

    separate_stems_output = tmp_path / "background.wav"
    separate_stems_output.write_bytes(b"fake")
    transcribe_output = tmp_path / "transcript.json"
    transcribe_output.write_text("{}", encoding="utf-8")
    translate_output = tmp_path / "translations.json"
    translate_output.write_text("{}", encoding="utf-8")
    synth_output = tmp_path / "synth_meta.json"
    synth_output.write_text("{}", encoding="utf-8")

    # Rodada 1: segmentação original, grava os hashes de todas as etapas.
    audio_hash = _hash_file(audio_path)
    vocals_hash = _hash_file(vocals_path)
    translate_jobs_v1 = [{"id": "seg0000", "source_text": "Hello there, how are you doing"}]
    translate_hash_v1 = _hash_json(translate_jobs_v1)
    synth_jobs_v1 = [{"id": "seg0000", "translated_text": "Oi, como vai"}]
    synth_hash_v1 = _hash_json(synth_jobs_v1)

    _record_stage_hash(tmp_path, "separate_stems", audio_hash)
    _record_stage_hash(tmp_path, "transcribe", vocals_hash)
    _record_stage_hash(tmp_path, "translate", translate_hash_v1)
    _record_stage_hash(tmp_path, "synthesize", synth_hash_v1)

    # Rodada 2: segmentação MUDOU (ex.: algoritmo de segment_words ajustado),
    # mesmo áudio/vocals. seg0000 agora cobre um trecho de texto diferente.
    translate_jobs_v2 = [{"id": "seg0000", "source_text": "Hello there, how are"}]
    translate_hash_v2 = _hash_json(translate_jobs_v2)
    synth_jobs_v2 = [{"id": "seg0000", "translated_text": "Oi, como"}]
    synth_hash_v2 = _hash_json(synth_jobs_v2)

    assert not _stage_needs_rerun(
        tmp_path, "separate_stems", audio_hash, separate_stems_output,
        force=False, force_stage=None,
    )
    assert not _stage_needs_rerun(
        tmp_path, "transcribe", vocals_hash, transcribe_output,
        force=False, force_stage=None,
    )
    assert _stage_needs_rerun(
        tmp_path, "translate", translate_hash_v2, translate_output,
        force=False, force_stage=None,
    )
    assert _stage_needs_rerun(
        tmp_path, "synthesize", synth_hash_v2, synth_output,
        force=False, force_stage=None,
    )
