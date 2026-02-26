"""Unit tests for agent_session_linker.plugins.registry.

Covers PluginRegistry, PluginNotFoundError, PluginAlreadyRegisteredError,
and the load_entrypoints entry-point discovery path (mocked via
importlib.metadata).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch

import pytest

from agent_session_linker.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)


# ---------------------------------------------------------------------------
# Shared base class and concrete implementations for test fixtures
# ---------------------------------------------------------------------------


class BasePlugin(ABC):
    @abstractmethod
    def run(self) -> str: ...


class AlphaPlugin(BasePlugin):
    def run(self) -> str:
        return "alpha"


class BetaPlugin(BasePlugin):
    def run(self) -> str:
        return "beta"


class GammaPlugin(BasePlugin):
    def run(self) -> str:
        return "gamma"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> PluginRegistry[BasePlugin]:
    """Fresh registry for each test."""
    return PluginRegistry(BasePlugin, "test-registry")


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class TestPluginNotFoundError:
    def test_is_key_error_subclass(self) -> None:
        err = PluginNotFoundError("missing", "my-registry")
        assert isinstance(err, KeyError)

    def test_plugin_name_attribute(self) -> None:
        err = PluginNotFoundError("missing", "my-registry")
        assert err.plugin_name == "missing"

    def test_registry_name_attribute(self) -> None:
        err = PluginNotFoundError("missing", "my-registry")
        assert err.registry_name == "my-registry"

    def test_message_contains_plugin_name(self) -> None:
        err = PluginNotFoundError("my-plugin", "reg")
        assert "my-plugin" in str(err)


class TestPluginAlreadyRegisteredError:
    def test_is_value_error_subclass(self) -> None:
        err = PluginAlreadyRegisteredError("dup", "my-registry")
        assert isinstance(err, ValueError)

    def test_plugin_name_attribute(self) -> None:
        err = PluginAlreadyRegisteredError("dup", "reg")
        assert err.plugin_name == "dup"

    def test_registry_name_attribute(self) -> None:
        err = PluginAlreadyRegisteredError("dup", "reg")
        assert err.registry_name == "reg"

    def test_message_contains_plugin_name(self) -> None:
        err = PluginAlreadyRegisteredError("dup-plugin", "reg")
        assert "dup-plugin" in str(err)


# ---------------------------------------------------------------------------
# PluginRegistry construction
# ---------------------------------------------------------------------------


class TestPluginRegistryConstruction:
    def test_initial_len_zero(self, registry: PluginRegistry[BasePlugin]) -> None:
        assert len(registry) == 0

    def test_list_plugins_empty_initially(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        assert registry.list_plugins() == []

    def test_repr_contains_registry_name(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        assert "test-registry" in repr(registry)

    def test_repr_contains_base_class_name(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        assert "BasePlugin" in repr(registry)


# ---------------------------------------------------------------------------
# register decorator
# ---------------------------------------------------------------------------


class TestPluginRegistryRegisterDecorator:
    def test_register_returns_class_unchanged(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        result = registry.register("alpha")(AlphaPlugin)
        assert result is AlphaPlugin

    def test_registered_class_retrievable(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register("alpha")(AlphaPlugin)
        assert registry.get("alpha") is AlphaPlugin

    def test_register_increments_len(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register("alpha")(AlphaPlugin)
        assert len(registry) == 1

    def test_register_duplicate_raises_already_registered(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register("alpha")(AlphaPlugin)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register("alpha")(BetaPlugin)

    def test_register_non_subclass_raises_type_error(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        class Unrelated:
            pass

        with pytest.raises(TypeError, match="subclass"):
            registry.register("bad")(Unrelated)  # type: ignore[arg-type]

    def test_register_logs_at_debug_level(
        self, registry: PluginRegistry[BasePlugin], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            registry.register("alpha")(AlphaPlugin)
        assert any("alpha" in r.message for r in caplog.records)

    def test_decorator_usage_style(self) -> None:
        reg: PluginRegistry[BasePlugin] = PluginRegistry(BasePlugin, "decorator-test")

        @reg.register("inline")
        class InlinePlugin(BasePlugin):
            def run(self) -> str:
                return "inline"

        assert reg.get("inline") is InlinePlugin


# ---------------------------------------------------------------------------
# register_class
# ---------------------------------------------------------------------------


class TestPluginRegistryRegisterClass:
    def test_register_class_stores_plugin(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register_class("beta", BetaPlugin)
        assert registry.get("beta") is BetaPlugin

    def test_register_class_duplicate_raises(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register_class("beta", BetaPlugin)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register_class("beta", AlphaPlugin)

    def test_register_class_non_subclass_raises_type_error(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        class Stranger:
            pass

        with pytest.raises(TypeError):
            registry.register_class("stranger", Stranger)  # type: ignore[arg-type]

    def test_register_class_logs_debug(
        self, registry: PluginRegistry[BasePlugin], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            registry.register_class("gamma", GammaPlugin)
        assert any("gamma" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# deregister
# ---------------------------------------------------------------------------


class TestPluginRegistryDeregister:
    def test_deregister_removes_plugin(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register_class("alpha", AlphaPlugin)
        registry.deregister("alpha")
        assert "alpha" not in registry

    def test_deregister_decrements_len(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register_class("alpha", AlphaPlugin)
        registry.deregister("alpha")
        assert len(registry) == 0

    def test_deregister_missing_raises_not_found(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        with pytest.raises(PluginNotFoundError):
            registry.deregister("ghost")

    def test_deregister_logs_debug(
        self, registry: PluginRegistry[BasePlugin], caplog: pytest.LogCaptureFixture
    ) -> None:
        registry.register_class("alpha", AlphaPlugin)
        with caplog.at_level(logging.DEBUG):
            registry.deregister("alpha")
        assert any("alpha" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestPluginRegistryGet:
    def test_get_returns_registered_class(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register_class("alpha", AlphaPlugin)
        assert registry.get("alpha") is AlphaPlugin

    def test_get_missing_raises_not_found(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        with pytest.raises(PluginNotFoundError, match="ghost"):
            registry.get("ghost")


# ---------------------------------------------------------------------------
# list_plugins
# ---------------------------------------------------------------------------


class TestPluginRegistryListPlugins:
    def test_list_plugins_returns_sorted_names(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register_class("gamma", GammaPlugin)
        registry.register_class("alpha", AlphaPlugin)
        registry.register_class("beta", BetaPlugin)
        assert registry.list_plugins() == ["alpha", "beta", "gamma"]

    def test_list_plugins_empty_initially(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        assert registry.list_plugins() == []


# ---------------------------------------------------------------------------
# __contains__ / __len__
# ---------------------------------------------------------------------------


class TestPluginRegistryContainsLen:
    def test_contains_true_after_register(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register_class("alpha", AlphaPlugin)
        assert "alpha" in registry

    def test_contains_false_before_register(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        assert "alpha" not in registry

    def test_contains_false_after_deregister(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        registry.register_class("alpha", AlphaPlugin)
        registry.deregister("alpha")
        assert "alpha" not in registry

    def test_len_reflects_count(self, registry: PluginRegistry[BasePlugin]) -> None:
        registry.register_class("alpha", AlphaPlugin)
        registry.register_class("beta", BetaPlugin)
        assert len(registry) == 2


# ---------------------------------------------------------------------------
# load_entrypoints
# ---------------------------------------------------------------------------


class TestPluginRegistryLoadEntrypoints:
    def _make_ep(self, name: str, cls: type) -> MagicMock:
        ep = MagicMock()
        ep.name = name
        ep.load.return_value = cls
        return ep

    def test_load_registers_valid_plugins(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        ep = self._make_ep("alpha", AlphaPlugin)
        with patch("importlib.metadata.entry_points", return_value=[ep]):
            registry.load_entrypoints("test.plugins")
        assert "alpha" in registry

    def test_load_skips_already_registered(
        self,
        registry: PluginRegistry[BasePlugin],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        registry.register_class("alpha", AlphaPlugin)
        ep = self._make_ep("alpha", AlphaPlugin)
        with patch("importlib.metadata.entry_points", return_value=[ep]):
            with caplog.at_level(logging.DEBUG):
                registry.load_entrypoints("test.plugins")
        # No duplicate error — skipped gracefully.
        assert len(registry) == 1

    def test_load_skips_plugin_that_fails_to_load(
        self,
        registry: PluginRegistry[BasePlugin],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ep = MagicMock()
        ep.name = "broken"
        ep.load.side_effect = ImportError("broken module")
        with patch("importlib.metadata.entry_points", return_value=[ep]):
            with caplog.at_level(logging.ERROR):
                registry.load_entrypoints("test.plugins")
        assert "broken" not in registry

    def test_load_skips_non_subclass_plugin(
        self,
        registry: PluginRegistry[BasePlugin],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class Incompatible:
            pass

        ep = self._make_ep("bad", Incompatible)  # type: ignore[arg-type]
        with patch("importlib.metadata.entry_points", return_value=[ep]):
            with caplog.at_level(logging.WARNING):
                registry.load_entrypoints("test.plugins")
        assert "bad" not in registry

    def test_load_idempotent_on_repeated_calls(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        ep = self._make_ep("alpha", AlphaPlugin)
        with patch("importlib.metadata.entry_points", return_value=[ep]):
            registry.load_entrypoints("test.plugins")
            registry.load_entrypoints("test.plugins")
        assert len(registry) == 1

    def test_load_multiple_plugins(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        eps = [
            self._make_ep("alpha", AlphaPlugin),
            self._make_ep("beta", BetaPlugin),
        ]
        with patch("importlib.metadata.entry_points", return_value=eps):
            registry.load_entrypoints("test.plugins")
        assert "alpha" in registry
        assert "beta" in registry
        assert len(registry) == 2

    def test_load_handles_empty_entrypoints(
        self, registry: PluginRegistry[BasePlugin]
    ) -> None:
        with patch("importlib.metadata.entry_points", return_value=[]):
            registry.load_entrypoints("test.plugins")
        assert len(registry) == 0
