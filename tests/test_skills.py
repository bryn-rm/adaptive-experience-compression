import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aec.config import AgentConfig, ALFWorldConfig, EmbeddingConfig, LLMConfig, MemoryConfig, MemoryWriterConfig
from aec.environments.alfworld import ALFWorldAdapter
from aec.experiments.artifacts import (
    BaselineConfig, MemoryWrite, RetrievalLog, TaskResult, create_manifest, load_baseline_config,
    read_rows,
)
from aec.experiments.baseline import run_baseline
from aec.experiments.comparison import compare_runs
from aec.llm import LLMResponse
from aec.memory.episodes import retrieval_text
from aec.memory.skills import (
    SkillParseError, build_skill, build_skill_writer_prompt, parse_skill, record_id, skill_section,
)
from aec.memory.store import MemoryStore, SkillStore
from aec.schemas import ProceduralSkill, TokenUsage
from aec.trajectories.schemas import AgentTrajectory
from test_memory import Embedder, count_tokens

SKILL = {"name": "Heat and place", "activation_conditions": ["Task asks to heat an object"],
         "procedure": ["Find the object", "Heat it with the microwave", "Place it"],
         "constraints": ["Hold the object while heating"],
         "failure_modes": ["Revisiting searched receptacles"]}
WRITER_PREFIX = "Write one reusable procedural skill"


class LLM:
    prompts = []
    outputs = []  # Writer outputs consumed in order; default is a valid skill.

    def generate(self, prompt, config):
        LLM.prompts.append(prompt)
        if prompt.startswith(WRITER_PREFIX):
            text = LLM.outputs.pop(0) if LLM.outputs else json.dumps(SKILL)
        else:
            text = "look"
        return LLMResponse(text=text, model_id=config.model_id, finish_reason="stop",
                           token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2))

    def close(self):
        pass


class Backend:
    def __init__(self, task_id):
        self.task_id, self.count = task_id, 0

    def infos(self):
        won = self.count == 2 and "fail" not in self.task_id
        return {"extra.gamefile": [self.task_id], "admissible_commands": [["look"]], "won": [won]}

    def reset(self):
        self.count = 0
        goal = Path(self.task_id).parent.parent.name.replace("-", " ")
        return [f"You are in a room.\n\nYour task is to: {goal}."], self.infos()

    def step(self, actions):
        self.count += 1
        return ["Nothing happens."], [float(self.infos()["won"][0])], [self.count == 2], self.infos()

    def close(self):
        pass


def skill_memory(tmp_path, **updates):
    values = dict(
        policy="skill", memory_format="skill_section_v1", store_path=tmp_path / "skills.sqlite",
        top_k=2, skill_grouping="task_family", skill_min_sources=2,
        embedding=EmbeddingConfig(model_id="fake-embedding"),
        writer=MemoryWriterConfig(prompt_version="skill_writer_v1",
                                  llm=LLMConfig(model_id="fake-writer", seed=3)),
    )
    return MemoryConfig(**(values | updates))


# Four heat tasks, two cool tasks, one failing clean task. Family = directory prefix.
NAMES = ["heat-apple", "heat-mug", "heat-egg", "heat-cup", "cool-mug", "cool-egg", "clean-fail"]


@pytest.fixture
def experiment(tmp_path):
    LLM.prompts, LLM.outputs, Embedder.calls = [], [], []
    backend = tmp_path / "backend.yaml"
    backend.write_text("env: {type: AlfredTWEnv}\n")
    candidates = []
    for name in NAMES:
        path = tmp_path / "games" / name / "trial" / "game.tw-pddl"
        path.parent.mkdir(parents=True)
        path.write_text(name)
        candidates.append(str(path))
    config = BaselineConfig(
        run_id="skill", manifest=tmp_path / "manifest.json", output_dir=tmp_path / "runs",
        environment=ALFWorldConfig(backend_config=backend, max_steps=3, seed=7),
        agent=AgentConfig(prompt_version="agent_v2", history_length=4,
                          llm=LLMConfig(model_id="fake", seed=11)),
        selection_count=len(NAMES), data_kind="fake_verification",
        memory=skill_memory(tmp_path),
    )
    manifest = create_manifest(config, tuple(candidates))

    def env_factory(settings, *, task_id):
        return ALFWorldAdapter(settings, backend=Backend(task_id), task_id=task_id)

    return config, manifest, env_factory


def run(config, factory, **kwargs):
    return run_baseline(config, env_factory=factory, llm_factory=LLM, embedder_factory=Embedder,
                        count_tokens=count_tokens, **kwargs)


def logs(config):
    root = config.output_dir / config.run_id
    return (read_rows(root / "trajectories.jsonl", AgentTrajectory),
            read_rows(root / "results.jsonl", TaskResult),
            read_rows(root / "retrievals.jsonl", RetrievalLog),
            read_rows(root / "memory_writes.jsonl", MemoryWrite))


def family(trajectory):
    return trajectory.task.metadata["task_family"]


# Skill format


@pytest.mark.parametrize("change", [
    {"extra": "x"}, {"name": " "}, {"name": 1}, {"procedure": []}, {"activation_conditions": []},
    {"procedure": "one step"}, {"constraints": ["ok", ""]}, {"failure_modes": [1]},
])
def test_malformed_skills_are_rejected(change):
    with pytest.raises(SkillParseError):
        parse_skill(json.dumps(SKILL | change))
    missing = {k: v for k, v in SKILL.items() if k != "constraints"}
    for text in ["", "not json", "[]", json.dumps(missing), "```\n" + json.dumps(SKILL) + "\n```"]:
        with pytest.raises(SkillParseError):
            parse_skill(text)


def test_skill_parsing_accepts_empty_optional_lists_and_one_json_fence():
    assert parse_skill(json.dumps(SKILL)) == SKILL
    assert parse_skill("```json\n" + json.dumps(SKILL) + "\n```") == SKILL
    sparse = SKILL | {"constraints": [], "failure_modes": []}
    assert parse_skill(json.dumps(sparse)) == sparse
    with pytest.raises(ValidationError):  # The boundary schema also rejects an empty procedure.
        ProceduralSkill(skill_id="s", source_trajectory_ids=["t"], embedding_text="x", version=1,
                        token_count=0, **(SKILL | {"procedure": []}))
    assert skill_section([]) is None


# Store


def test_skill_store_versions_latest_exclusion_and_kind(tmp_path):
    config = EmbeddingConfig(model_id="e")
    db = SkillStore(tmp_path / "s.sqlite", embedding=config, run_id="r")

    def skill(skill_id, version, sources):
        return ProceduralSkill(skill_id=skill_id, version=version, source_trajectory_ids=sources,
                               embedding_text="x", token_count=1, **SKILL)

    db.add(skill("skill-heat", 1, ["t1", "t2"]), source_task_ids=["a", "b"], embedding=[1.0, 0.0])
    db.add(skill("skill-cool", 1, ["t3", "t4"]), source_task_ids=["c", "d"], embedding=[0.0, 1.0])
    db.add(skill("skill-heat", 2, ["t1", "t2", "t5"]), source_task_ids=["a", "b", "e"],
           embedding=[1.0, 0.2])
    for bad in [skill("skill-heat", 2, ["t9", "t8"]), skill("skill-cool", 3, ["t9", "t8"])]:
        with pytest.raises(ValueError, match="contiguous"):
            db.add(bad, source_task_ids=["x", "y"], embedding=[1.0, 0.0])
    with pytest.raises(ValueError):
        db.add(skill("skill-x", 1, ["t9", "t8"]), source_task_ids=["x"], embedding=[1.0, 0.0])
    assert len(db.skills()) == 3
    assert [record_id(s.skill) for s in db.latest()] == ["skill-cool@v1", "skill-heat@v2"]
    eligible, top = db.retrieve([1.0, 0.0], top_k=5)
    assert eligible == 2 and [record_id(s.skill) for s, _ in top] == ["skill-heat@v2", "skill-cool@v1"]
    eligible, top = db.retrieve([1.0, 0.0], top_k=5, exclude_task_id="e")
    assert eligible == 1 and [record_id(s.skill) for s, _ in top] == ["skill-cool@v1"]
    db.close()
    reopened = SkillStore(tmp_path / "s.sqlite", embedding=config, run_id=None, read_only=True)
    assert len(reopened.skills()) == 3
    reopened.close()
    with pytest.raises(ValueError, match="memory"):
        MemoryStore(tmp_path / "s.sqlite", embedding=config, run_id="r")


# Full skill policy runs


def test_skills_follow_fixed_rule_with_full_provenance_and_no_leakage(experiment):
    config, manifest, factory = experiment
    metrics = run(config, factory)
    trajectories, results, retrievals, writes = logs(config)
    order = list(manifest.task_ids)
    by_id = {t.trajectory_id: t for t in trajectories}
    stored = [w for w in writes if w.status == "stored"]
    # A family's k-th success (k >= 2) triggers version k-1 built from successes 1..k.
    expected = []
    for t in trajectories:
        prior = [p for p in trajectories[:trajectories.index(t) + 1]
                 if p.success and family(p) == family(t)]
        if t.success and len(prior) >= 2:
            expected.append((t.trajectory_id, [p.trajectory_id for p in prior], len(prior) - 1))
    assert [(w.trajectory_id, w.skill.source_trajectory_ids, w.skill.version) for w in stored] == expected
    assert len(stored) == 4  # heat: 3 versions, cool: 1, clean: none.
    for w in stored:
        sources = [by_id[s] for s in w.skill.source_trajectory_ids]
        assert w.skill.skill_id == f"skill-{family(sources[0])}"
        assert w.source_task_ids == [s.task_id for s in sources]
        assert w.skill.embedding_text == "\n".join(retrieval_text(s.task, "task_goal") for s in sources)
        assert w.prompt == build_skill_writer_prompt(sources)
        assert all(s.success and family(s) == family(sources[0]) for s in sources)
    # No future leakage: every retrieved skill version was built only from earlier tasks,
    # and it is the latest version of that skill at retrieval time.
    versions = {record_id(w.skill): w for w in stored}
    for row in retrievals:
        here = order.index(row.task_id)
        for record, sources in zip(row.records, row.source_trajectory_ids):
            w = versions[record.memory_id]
            assert sources == w.skill.source_trajectory_ids
            assert all(order.index(by_id[s].task_id) < here for s in sources)
            newer = [v for v in stored if v.skill.skill_id == w.skill.skill_id
                     and v.skill.version > w.skill.version]
            assert all(order.index(v.task_id) >= here for v in newer)
        for step in next(t for t in trajectories if t.task_id == row.task_id).steps:
            assert (row.memory_section or "") in step.prompt
            assert (row.memory_section is None) == ("Relevant skills" not in step.prompt)
    assert [r.retrieved_memory_ids for r in results] == [[x.memory_id for x in r.records]
                                                          for r in retrievals]
    memory = metrics["memory"]
    assert memory["policy"] == "skill" and memory["writes_stored"] == 4
    assert memory["skill_versions_stored"] == 4 and memory["store_skills"] == 2
    assert memory["retrieved_memories"] == sum(len(r.records) for r in retrievals)


def test_rejected_synthesis_stores_nothing_and_next_success_retries(experiment):
    config, _, factory = experiment
    bad = json.dumps(SKILL | {"procedure": []})
    LLM.outputs = [bad, bad]
    metrics = run(config, factory)
    trajectories, _, _, writes = logs(config)
    finished = [w for w in writes if w.status != "started"]
    assert [w.status for w in finished[:2]] == ["rejected", "rejected"]
    assert all(w.skill is None and "procedure" in w.error for w in finished[:2])
    stored = [w for w in finished if w.status == "stored"]
    # Versions stay contiguous; the surviving heat skill still uses every success so far.
    heat = [w for w in stored if w.skill.skill_id == "skill-heat"]
    assert [w.skill.version for w in heat] == list(range(1, len(heat) + 1))
    assert len(heat[-1].skill.source_trajectory_ids) == 4
    assert metrics["memory"]["writes_rejected"] == 2


def test_skill_resume_is_idempotent_and_rebuilds_store(experiment):
    config, _, factory = experiment
    metrics = run(config, factory)
    root = config.output_dir / config.run_id
    before = {p: p.read_bytes() for p in root.glob("*.json*")}
    calls = len(LLM.prompts)

    def forbidden(*args, **kwargs):
        pytest.fail("Resume must not act for finished tasks")

    assert run(config, forbidden) == metrics
    assert len(LLM.prompts) == calls
    config.memory.store_path.unlink()
    assert run(config, forbidden) == metrics
    db = SkillStore(config.memory.store_path, embedding=config.memory.embedding, run_id=None,
                    read_only=True)
    assert [s.skill for s in db.skills()] == [w.skill for w in logs(config)[3] if w.status == "stored"]
    assert all(p.read_bytes() == v for p, v in before.items())


def test_skill_resume_rejects_missing_synthesis(experiment):
    config, _, factory = experiment
    run(config, factory)
    path = config.output_dir / config.run_id / "memory_writes.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-2]) + "\n")  # Drop the last synthesis.
    config.memory.store_path.unlink()
    with pytest.raises(ValueError, match="skill synthesis"):
        run(config, factory)


def test_skill_and_episode_runs_share_agent_and_compare(experiment, tmp_path):
    from test_memory import memory_config
    config, manifest, factory = experiment
    run(config, factory)
    skill_trajectories = logs(config)[0]
    config.run_id, config.memory = "episodic", memory_config(tmp_path / "ep")
    run(config, factory)
    episode_trajectories = logs(config)[0]
    for a, b in zip(skill_trajectories, episode_trajectories):
        assert (a.task_id, a.agent_config, a.environment_config) == (
            b.task_id, b.agent_config, b.environment_config)
    comparison = compare_runs(config.output_dir / "episodic", config.output_dir / "skill")
    assert (comparison["previous_policy"], comparison["candidate_policy"]) == ("episodic", "skill")
    assert "identical acting setup" in comparison["interpretation"]


@pytest.mark.parametrize("updates", [
    {"memory_format": "memory_section_v1"},
    {"writer": MemoryWriterConfig(prompt_version="episode_writer_v1", llm=LLMConfig(model_id="w"))},
    {"skill_min_sources": None}, {"skill_grouping": None}, {"skill_min_sources": 1},
])
def test_skill_config_must_be_consistent(tmp_path, updates):
    with pytest.raises(ValidationError):
        skill_memory(tmp_path, **updates)


def test_episodic_config_rejects_skill_settings(tmp_path):
    from test_memory import memory_config
    with pytest.raises(ValidationError):
        memory_config(tmp_path, skill_min_sources=2)


def test_skill_pilot_reuses_retrieval_machinery_and_acting_setup():
    skill = load_baseline_config("configs/skill_pilot.yaml")
    episodic = load_baseline_config("configs/episodic_pilot.yaml")
    baseline = load_baseline_config("configs/baseline_calibration.yaml")
    assert (skill.agent, skill.environment, skill.manifest) == (
        baseline.agent, baseline.environment, baseline.manifest)
    shared = ("embedding", "top_k", "retrieval_key", "exclude_same_task", "tokenizer")
    assert all(getattr(skill.memory, f) == getattr(episodic.memory, f) for f in shared)
    assert skill.memory.writer.llm == episodic.memory.writer.llm
    assert skill.memory.store_path != episodic.memory.store_path
    assert (skill.memory.skill_grouping, skill.memory.skill_min_sources) == ("task_family", 2)
    assert len({skill.run_id, episodic.run_id, baseline.run_id}) == 3
