"""The LTX LoRA types round-trip through the type registry."""

from nodetool.metadata.types import HFLoraLTX, HFLoraLTXConfig, NameToType


def test_lora_ltx_config_round_trips_by_type_name():
    config = HFLoraLTXConfig(
        lora=HFLoraLTX(repo_id="someone/ltx-character", path="character.safetensors"),
        strength=0.8,
    )
    data = config.model_dump()

    assert data["type"] == "hf.lora_ltx_config"
    assert data["lora"]["type"] == "hf.lora_ltx"
    restored = NameToType["hf.lora_ltx_config"](**data)
    assert restored == config
    assert HFLoraLTXConfig().strength == 1.0
    assert not HFLoraLTXConfig().lora.is_set()
