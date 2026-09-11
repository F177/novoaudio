def test_packages_importable() -> None:
    import packages.pipeline
    import packages.ptbr

    assert packages.pipeline is not None
    assert packages.ptbr is not None
