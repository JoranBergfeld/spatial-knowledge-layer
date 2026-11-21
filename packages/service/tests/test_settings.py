import pytest

from spatial_service.app import ConfigError, Settings

BASE = {
    "SPATIAL_SERVICE_DATABASE_URL": "postgresql://spatial_service@db/spatial",
    "SPATIAL_PLANNING_DEPLOYMENT": "planner",
    "SPATIAL_COMPOSITION_DEPLOYMENT": "composer",
}
AZURE = {"AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com/", "AZURE_OPENAI_API_KEY": "k"}
SPARK = {"SPATIAL_LLM_BASE_URL": "http://192.168.5.6:8101/v1", "SPATIAL_LLM_API_KEY": "s"}
ONE_MODEL = {
    "SPATIAL_PLANNING_DEPLOYMENT": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "SPATIAL_COMPOSITION_DEPLOYMENT": "nvidia/Qwen3.8-Flash-Next-NVFP4",
}


def test_azure_needs_its_endpoint_and_key() -> None:
    with pytest.raises(ConfigError, match="AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY"):
        Settings.from_env(BASE)


def test_azure_needs_two_deployments() -> None:
    same = {**BASE, **AZURE, "SPATIAL_COMPOSITION_DEPLOYMENT": "planner"}
    with pytest.raises(ConfigError, match="separate deployments"):
        Settings.from_env(same)


def test_a_v1_server_needs_no_azure() -> None:
    settings = Settings.from_env({**BASE, **SPARK})
    assert (settings.llm_base_url, settings.llm_api_key) == (SPARK["SPATIAL_LLM_BASE_URL"], "s")
    assert settings.endpoint == ""


def test_a_v1_server_needs_its_key() -> None:
    with pytest.raises(ConfigError, match="SPATIAL_LLM_API_KEY"):
        Settings.from_env({**BASE, "SPATIAL_LLM_BASE_URL": SPARK["SPATIAL_LLM_BASE_URL"]})


def test_one_model_on_a_v1_server_may_plan_and_compose() -> None:
    settings = Settings.from_env({**BASE, **SPARK, **ONE_MODEL})
    assert settings.planning_deployment == settings.composition_deployment


def test_azure_is_used_when_no_v1_server_is_named() -> None:
    assert Settings.from_env({**BASE, **AZURE}).llm_base_url is None
