from scripts.dub import build_stage_report, words_from_whisperx_transcript


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
        },
        {
            "translated_ok": True,
            "synthesized_ok": False,
            "within_duration_tolerance": False,
            "evaluated_ok": False,
            "any_flag": True,
        },
    ]
    report = build_stage_report(results)
    assert report["total_segments"] == 2
    assert report["translated_rate"] == 1.0
    assert report["synthesized_rate"] == 0.5
    assert report["within_duration_tolerance_rate"] == 0.5
    assert report["evaluated_rate"] == 0.5
    assert report["segments_with_any_flag_rate"] == 0.5
