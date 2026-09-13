from scripts.dub import _is_bad_synthesis, build_stage_report, words_from_whisperx_transcript


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
