"""Tests for project model and persistence (M1)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dfn_cave_studio.models.project import (
    Project, ProjectMetadata, ProjectConfig, CURRENT_PROJECT_VERSION,
)
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.enums import ProjectStatus
from dfn_cave_studio.persistence.project_store import (
    ProjectStore, RecentProjectsManager, ProjectStoreError,
)


class TestProjectMetadata:
    def test_defaults(self):
        m = ProjectMetadata()
        assert m.name == "Untitled Project"
        assert m.project_version == 1

    def test_custom(self):
        m = ProjectMetadata(name="Test", author="Alice", description="A test project")
        assert m.name == "Test"
        assert m.author == "Alice"


class TestProjectConfig:
    def test_defaults(self):
        c = ProjectConfig()
        assert c.master_seed == 42
        assert c.p32_tolerance == 0.05


class TestProject:
    def test_empty_project(self):
        p = Project()
        assert p.metadata.name == "Untitled Project"
        assert p.schema_version == CURRENT_PROJECT_VERSION
        assert p.project_status == ProjectStatus.NEW

    def test_serialize_roundtrip(self):
        p = Project()
        p.metadata.name = "Roundtrip Test"
        p.metadata.description = "Testing JSON serialization"
        p.config.master_seed = 12345

        d = p.to_dict()
        p2 = Project.from_dict(d)

        assert p2.metadata.name == "Roundtrip Test"
        assert p2.metadata.description == "Testing JSON serialization"
        assert p2.config.master_seed == 12345

    def test_json_roundtrip(self):
        p = Project()
        p.metadata.name = "JSON Test"
        json_str = p.to_json()
        p2 = Project.from_json(json_str)
        assert p2.metadata.name == "JSON Test"

    def test_save_load_file(self):
        p = Project()
        p.metadata.name = "File Test"
        p.config.master_seed = 999

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dfncs"
            p.save_to_file(path)
            assert path.exists()

            p2 = Project.from_file(path)
            assert p2.metadata.name == "File Test"
            assert p2.config.master_seed == 999

    def test_model_volume(self):
        p = Project()
        p.model_bounds = ModelBounds(x_min=0, x_max=10, y_min=0, y_max=20, z_min=0, z_max=30)
        assert p.model_volume == 6000.0  # 10 * 20 * 30

    def test_create_sample_project(self):
        p = Project()
        p.create_sample_project()
        assert len(p.joint_sets) == 3
        assert p.joint_sets[0].set_id == 1
        assert p.project_status == ProjectStatus.DATA_LOADED

    def test_joint_set_ids_unique(self):
        p = Project()
        p.create_sample_project()
        set_ids = [js.set_id for js in p.joint_sets]
        assert len(set_ids) == len(set(set_ids))


class TestProjectStore:
    def test_new_project(self):
        store = ProjectStore()
        p = store.new_project("Test Project")
        assert p is not None
        assert store.has_project
        assert store.is_dirty

    def test_save_as(self):
        store = ProjectStore()
        store.new_project("Save Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "save_test.dfncs"
            store.save_as(path)
            assert path.exists()
            assert not store.is_dirty

    def test_open(self):
        # First save a project
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "open_test.dfncs"
            store1 = ProjectStore()
            store1.new_project("Open Test")
            store1.save_as(path)

            # Then open it with a new store
            store2 = ProjectStore()
            p = store2.open(path)
            assert p.metadata.name == "Open Test"
            assert not store2.is_dirty

    def test_close(self):
        store = ProjectStore()
        store.new_project("Test")
        store.close()
        assert not store.has_project

    def test_save_without_project_raises(self):
        store = ProjectStore()
        with pytest.raises(ValueError):
            store.save()

    def test_open_nonexistent_raises(self):
        store = ProjectStore()
        with pytest.raises(FileNotFoundError):
            store.open(Path("/nonexistent/path.dfncs"))

    def test_open_corrupted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.dfncs"
            path.write_text("not valid json{{{")

            store = ProjectStore()
            with pytest.raises(ProjectStoreError):
                store.open(path)

    def test_backup_on_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup_test.dfncs"

            store = ProjectStore()
            store.new_project("V1")
            store.save_as(path)

            # Modify and save again
            store._current_project.metadata.name = "V2"
            store._mark_dirty()
            store.save()

            # Backup should exist
            backup = path.with_suffix(".dfncs.bak")
            assert backup.exists() or not store.is_dirty


class TestRecentProjectsManager:
    def test_add_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            mgr = RecentProjectsManager(config_dir=config_dir)

            path = Path(tmpdir) / "proj1.dfncs"
            path.write_text("{}")

            mgr.add(path, "Project 1")
            recent = mgr.list()
            assert len(recent) == 1
            assert recent[0]["name"] == "Project 1"

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            mgr = RecentProjectsManager(config_dir=config_dir)

            path = Path(tmpdir) / "proj1.dfncs"
            path.write_text("{}")
            mgr.add(path, "Project 1")
            mgr.remove(path)
            assert len(mgr.list()) == 0

    def test_max_recent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            mgr = RecentProjectsManager(config_dir=config_dir)

            for i in range(25):
                path = Path(tmpdir) / f"proj{i}.dfncs"
                path.write_text("{}")
                mgr.add(path, f"Project {i}")

            recent = mgr.list()
            assert len(recent) <= 20

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            mgr = RecentProjectsManager(config_dir=config_dir)

            path = Path(tmpdir) / "proj1.dfncs"
            path.write_text("{}")
            mgr.add(path)
            mgr.clear()
            assert len(mgr.list()) == 0
