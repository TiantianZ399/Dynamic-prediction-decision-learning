def test_imports():
    import dynport
    from dynport.deep_cql_fqi import DeepCQLConfig
    assert dynport.__version__
    assert DeepCQLConfig().train_len == 250
