from scripts import dump_torch_model_structure


def test_dump_torch_model_structure_builds_dummy_payload():
    args = dump_torch_model_structure.Args(config="dummy", include_gemma=True, include_siglip=True)
    config = dump_torch_model_structure._model_config(args)
    model = dump_torch_model_structure.ACOTVLATorch(config)
    summary = dump_torch_model_structure._state_dict_summary(model)

    assert model.model_status == "torch_acot_gemma_siglip_prefix"
    assert any(item["name"].startswith("paligemma_llm.") for item in summary)
    assert any(item["name"].startswith("paligemma_img.") for item in summary)
    assert all("shape" in item for item in summary)
