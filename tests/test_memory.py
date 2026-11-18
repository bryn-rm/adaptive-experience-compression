import json
from pathlib import Path
import sqlite3

import pytest

from aec.agents.v2 import ObservationAction, build_prompt_v2
from aec.config import AgentConfig, ALFWorldConfig, EmbeddingConfig, EpisodeWriterConfig, LLMConfig, MemoryConfig
from aec.environments.alfworld import ALFWorldAdapter
from aec.experiments.artifacts import (
    BaselineConfig, MemoryWrite, RetrievalLog, TaskResult, create_manifest, load_baseline_config,
    read_rows,
)
from aec.experiments.baseline import run_baseline
from aec.llm import LLMResponse
from aec.memory.embeddings import EmbeddingResponse, OpenAIEmbedder, cosine
from aec.memory.episodes import (
    EpisodeParseError, build_episode, build_writer_prompt, memory_section, parse_episode,
    retrieval_text,
)
from aec.memory.store import MemoryStore
from aec.schemas import EpisodicMemory, Task, TokenUsage
from aec.trajectories.schemas import AgentTrajectory

EPISODE = {"task_summary": "Put an object in a drawer.",
           "experience": "Took it from the counter and put it in drawer 1.",
           "outcome": "Succeeded."}
WRITER_PREFIX = "Write a concise episodic memory"


def count_tokens(text):
    return len(text.split())


class Embedder:
    calls = []

    def embed(self, texts, config):
        Embedder.calls.extend(texts)
        # Deterministic letter histogram; similar goals give similar vectors.
        vectors = [[1.0 + text.count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for text in texts]
        return EmbeddingResponse(vectors=vectors, model_id=config.model_id,
                                 prompt_tokens=sum(len(t.split()) for t in texts))

    def close(self):
        pass


class LLM:
    prompts = []
    writer_output = json.dumps(EPISODE)

    def generate(self, prompt, config):
        LLM.prompts.append(prompt)
        text = LLM.writer_output if prompt.startswith(WRITER_PREFIX) else "look"
        return LLMResponse(text=text, model_id=config.model_id, finish_reason="stop",
                           token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2))

    def close(self):
        pass


def description(task_id):
    name = Path(task_id).name
    return f"You are in a room.\n\nYour task is to: put some {name.split('-')[1]} in drawer."


class Backend:
    def __init__(self, task_id):
        self.task_id, self.count = task_id, 0

    def infos(self):
        won = self.count == 2 and not Path(self.task_id).name.startswith("fail")
        return {"extra.gamefile": [self.task_id], "admissible_commands": [["look"]], "won": [won]}

    def reset(self):
        self.count = 0
        return [description(self.task_id)], self.infos()

    def step(self, actions):
        self.count += 1
        return ["Nothing happens."], [float(self.infos()["won"][0])], [self.count == 2], self.infos()

    def close(self):
        pass


def memory_config(tmp_path, **updates):
    return MemoryConfig(
        store_path=tmp_path / "memory" / "episodes.sqlite", top_k=2,
        embedding=EmbeddingConfig(model_id="fake-embedding"),
        writer=EpisodeWriterConfig(llm=LLMConfig(model_id="fake-writer", seed=3)), **updates,
    )


@pytest.fixture
def experiment(tmp_path):
    LLM.prompts, Embedder.calls, LLM.writer_output = [], [], json.dumps(EPISODE)
    backend = tmp_path / "backend.yaml"
    backend.write_text("env: {type: AlfredTWEnv}\n")
    candidates = []
    for name in ("ok-mug", "ok-cup", "fail-plate", "ok-mugs"):
        path = tmp_path / name
        path.write_text(name)
        candidates.append(str(path))
    config = BaselineConfig(
        run_id="episodic", manifest=tmp_path / "manifest.json", output_dir=tmp_path / "runs",
        environment=ALFWorldConfig(backend_config=backend, max_steps=3, seed=7),
        agent=AgentConfig(prompt_version="agent_v2", history_length=4,
                          llm=LLMConfig(model_id="fake", seed=11)),
        selection_count=4, data_kind="fake_verification", memory=memory_config(tmp_path),
    )
    manifest = create_manifest(config, tuple(candidates))

    def env_factory(settings, *, task_id):
        return ALFWorldAdapter(settings, backend=Backend(task_id), task_id=task_id)

    return config, manifest, env_factory


def run(config, factory, **kwargs):
    return run_baseline(config, env_factory=factory, llm_factory=kwargs.pop("llm", LLM),
                        embedder_factory=kwargs.pop("embedder_factory", Embedder),
                        count_tokens=count_tokens, **kwargs)


def logs(config):
    root = config.output_dir / config.run_id
    return (read_rows(root / "trajectories.jsonl", AgentTrajectory),
            read_rows(root / "results.jsonl", TaskResult),
            read_rows(root / "retrievals.jsonl", RetrievalLog),
            read_rows(root / "memory_writes.jsonl", MemoryWrite))


def store(config):
    return MemoryStore(config.memory.store_path, embedding=config.memory.embedding,
                       run_id=None, read_only=True)


# Prompt and episode formatting


def test_no_memory_section_leaves_agent_v2_prompt_unchanged():
    task = Task(task_id="t", description="Put mug on table")
    history = [ObservationAction("A mug", "take mug")]
    base = build_prompt_v2(task, "Holding mug", ["look"], history, history_length=4)
    assert build_prompt_v2(task, "Holding mug", ["look"], history, history_length=4,
                           memory_section=None) == base
    assert memory_section([]) is None
    memory = EpisodicMemory(memory_id="m", source_trajectory_ids=["t0"], embedding_text="x",
                            token_count=1, **EPISODE)
    section = memory_section([memory])
    prompt = build_prompt_v2(task, "Holding mug", ["look"], history, history_length=4,
                             memory_section=section)
    lines = prompt.splitlines(keepends=True)
    assert lines[3] == section and "".join(lines[:3] + lines[4:]) == base
    assert json.loads(section.split(": ", 1)[1]) == [
        {"task": EPISODE["task_summary"], "experience": EPISODE["experience"],
         "outcome": EPISODE["outcome"]}]
    for bad in ["", "no newline", "two\nlines\n"]:
        with pytest.raises(ValueError):
            build_prompt_v2(task, "o", [], [], history_length=0, memory_section=bad)


def test_retrieval_key_is_deterministic_goal_text():
    task = Task(task_id="t", description="Room with a drawer.\n\nYour task is to: put a mug in drawer.")
    assert retrieval_text(task, "task_goal") == "put a mug in drawer."
    assert retrieval_text(task, "task_description") == task.description
    for text in ["No goal here", "Your task is to: a\nYour task is to: b", "Your task is to: \n"]:
        with pytest.raises(ValueError):
            retrieval_text(Task(task_id="t", description=text), "task_goal")


@pytest.mark.parametrize("text", [
    "", "not json", "[]", json.dumps({**EPISODE, "extra": "x"}),
    json.dumps({k: v for k, v in EPISODE.items() if k != "outcome"}),
    json.dumps({**EPISODE, "outcome": " "}), json.dumps({**EPISODE, "outcome": 1}),
    "```\n" + json.dumps(EPISODE) + "\n```", "Here: " + json.dumps(EPISODE),
])
def test_malformed_episodes_are_rejected(text):
    with pytest.raises(EpisodeParseError):
        parse_episode(text)


def test_episode_parsing_accepts_only_exact_json_or_one_json_fence():
    assert parse_episode(json.dumps(EPISODE)) == EPISODE
    assert parse_episode("```json\n" + json.dumps(EPISODE) + "\n```") == EPISODE


# Store and retrieval


def test_cosine_and_store_ordering_exclusion_persistence(tmp_path):
    config = EmbeddingConfig(model_id="e")
    path = tmp_path / "s.sqlite"
    db = MemoryStore(path, embedding=config, run_id="r")

    def memory(i):
        return EpisodicMemory(memory_id=f"m{i}", source_trajectory_ids=[f"t{i}"],
                              embedding_text="x", token_count=1, **EPISODE)

    db.add(memory(1), source_task_id="a", embedding=[1.0, 0.0])
    db.add(memory(2), source_task_id="b", embedding=[0.0, 1.0])
    db.add(memory(3), source_task_id="c", embedding=[2.0, 0.0])  # Ties m1 on similarity.
    eligible, top = db.retrieve([1.0, 0.1], top_k=2)
    assert eligible == 3 and [e.memory.memory_id for e, _ in top] == ["m1", "m3"]
    assert top[0][1] == top[1][1] > 0.99
    eligible, top = db.retrieve([1.0, 0.1], top_k=5, exclude_task_id="a")
    assert eligible == 2 and [e.memory.memory_id for e, _ in top] == ["m3", "m2"]
    with pytest.raises(sqlite3.IntegrityError):
        db.add(memory(1), source_task_id="a", embedding=[1.0, 0.0])
    db.close()
    reopened = MemoryStore(path, embedding=config, run_id=None, read_only=True)
    assert [e.memory.memory_id for e in reopened.episodes()] == ["m1", "m2", "m3"]
    with pytest.raises(ValueError):
        reopened.add(memory(4), source_task_id="d", embedding=[1.0, 0.0])
    reopened.close()
    with pytest.raises(ValueError, match="embedding"):
        MemoryStore(path, embedding=EmbeddingConfig(model_id="other"), run_id="r")
    with pytest.raises(ValueError, match="owner"):
        MemoryStore(path, embedding=config, run_id="another-run")
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == -1
    for a, b in [([1.0], [1.0, 0.0]), ([0.0], [1.0])]:
        with pytest.raises(ValueError):
            cosine(a, b)


def test_openai_embedder_adapter_preserves_order_and_usage():
    class Item:
        def __init__(self, index, embedding):
            self.index, self.embedding = index, embedding

    class Response:
        model = "text-embedding-3-small"
        data = [Item(1, [0.0, 1.0]), Item(0, [1.0, 0.0])]

        class usage:
            prompt_tokens = 7

    class Client:
        def __init__(self):
            self.embeddings = self

        def create(self, **kwargs):
            self.kwargs = kwargs
            return Response()

    client = Client()
    config = EmbeddingConfig(model_id="text-embedding-3-small")
    result = OpenAIEmbedder(client=client).embed(["a", "b"], config)
    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]] and result.prompt_tokens == 7
    assert "dimensions" not in client.kwargs and client.kwargs["input"] == ["a", "b"]
    with pytest.raises(ValueError):
        OpenAIEmbedder(client=client).embed([""], config)


# Full episodic policy runs


def test_episodic_stream_writes_retrieves_and_preserves_provenance(experiment):
    config, manifest, factory = experiment
    metrics = run(config, factory)
    trajectories, results, retrievals, writes = logs(config)
    order = list(manifest.task_ids)
    by_task = {t.task_id: t for t in trajectories}
    assert [r.task_id for r in retrievals] == order
    stored = [w for w in writes if w.status == "stored"]
    successes = [t for t in trajectories if t.success]
    assert len(stored) == len(successes) == 3
    for write in stored:
        source = by_task[write.task_id]
        assert write.memory.source_trajectory_ids == [source.trajectory_id] and source.success
        assert write.memory.embedding_text == retrieval_text(source.task, "task_goal")
        assert write.prompt == build_writer_prompt(source)
    # No future leakage: retrieved memories come only from earlier, successful tasks.
    episode_source = {w.memory.memory_id: w.task_id for w in stored}
    for row in retrievals:
        for record in row.records:
            assert order.index(episode_source[record.memory_id]) < order.index(row.task_id)
            assert by_task[episode_source[record.memory_id]].success
    assert not retrievals[0].records and retrievals[0].memory_section is None
    assert retrievals[-1].records  # Earlier successes exist for the last task.
    # The injected section is exactly what the agent saw on every step.
    for row in retrievals:
        for step in by_task[row.task_id].steps:
            assert (row.memory_section or "") in step.prompt
            assert (row.memory_section is None) == ("Relevant past experiences" not in step.prompt)
    assert all(r.retrieved_memory_ids == [x.memory_id for x in q.records]
               for r, q in zip(results, retrievals))
    db = store(config)
    assert [e.memory for e in db.episodes()] == [w.memory for w in stored]
    db.close()
    memory = metrics["memory"]
    assert memory["episodes_stored"] == memory["store_episodes"] == 3
    assert memory["retrieval_logs"] == 4
    assert memory["retrieved_memories"] == sum(len(r.records) for r in retrievals)
    assert memory["injected_memory_tokens"] == sum(r.memory_tokens for r in retrievals) > 0
    assert memory["writer_prompt_tokens"] == 30 and memory["writer_usage_known"]
    # Query calls only when something is retrievable; one embedding per stored episode.
    assert len(Embedder.calls) == 3 + sum(r.eligible_count > 0 for r in retrievals)
    resolved = json.loads((config.output_dir / config.run_id / "resolved_config.json").read_text())
    assert resolved["policy"] == "episodic" and resolved["config"]["memory"]["top_k"] == 2


def test_none_and_episodic_policies_share_the_base_agent(experiment):
    config, manifest, factory = experiment
    run(config, factory)
    episodic, _, retrievals, _ = logs(config)
    LLM.prompts = []
    config.run_id, config.memory = "none", None
    metrics = run(config, factory)
    plain, _, _, _ = logs(config)
    assert "memory" not in metrics
    assert not any(p.startswith(WRITER_PREFIX) for p in LLM.prompts)
    assert not (config.output_dir / "none" / "retrievals.jsonl").exists()
    for e, p, r in zip(episodic, plain, retrievals):
        assert e.agent_config == p.agent_config and e.environment_config == p.environment_config
        assert e.task_id == p.task_id
        section = r.memory_section or ""
        assert [s.prompt.replace(section, "") for s in e.steps] == [s.prompt for s in p.steps]


def test_writer_rejection_and_failure_store_nothing_and_continue(experiment):
    config, _, factory = experiment
    LLM.writer_output = "Sure! " + json.dumps(EPISODE)
    metrics = run(config, factory)
    _, results, retrievals, writes = logs(config)
    assert len(results) == 4 and metrics["successful_tasks"] == 3
    assert [w.status for w in writes] == ["started", "rejected"] * 3
    assert all(w.output.startswith("Sure!") and w.error for w in writes if w.status == "rejected")
    assert not any(r.records for r in retrievals) and metrics["memory"]["store_episodes"] == 0
    assert metrics["memory"]["writes_rejected"] == 3


def test_resume_is_idempotent_and_rebuilds_store_from_log(experiment):
    config, _, factory = experiment
    metrics = run(config, factory)
    root = config.output_dir / config.run_id
    before = {p: p.read_bytes() for p in root.glob("*.json*")}
    calls = (len(LLM.prompts), len(Embedder.calls))
    def forbidden(*args, **kwargs):
        pytest.fail("Resume must not act for finished tasks")
    assert run(config, forbidden) == metrics
    assert (len(LLM.prompts), len(Embedder.calls)) == calls
    assert all(p.read_bytes() == v for p, v in before.items())
    # The store is an index of the append-only log and can be rebuilt from it.
    config.memory.store_path.unlink()
    assert run(config, forbidden) == metrics
    assert len(store(config).episodes()) == 3


def test_resume_rejects_tampered_or_foreign_store(experiment, tmp_path):
    config, _, factory = experiment
    run(config, factory)
    with sqlite3.connect(config.memory.store_path) as db:
        db.execute("UPDATE episodes SET memory = replace(memory, 'drawer 1', 'drawer 2')")
    with pytest.raises(ValueError, match="differs"):
        run(config, factory)
    config.run_id = "fresh-run"
    with pytest.raises(ValueError, match="predates"):
        run(config, factory)
    assert not (config.output_dir / "fresh-run" / "attempts.jsonl").exists()


def test_interrupted_episode_write_is_not_replayed(experiment, monkeypatch):
    import aec.experiments.memory as module
    config, manifest, factory = experiment
    original = module.MemoryStore.add
    def crash(self, *args, **kwargs):
        raise KeyboardInterrupt
    class Crash(LLM):
        def generate(self, prompt, settings):
            if prompt.startswith(WRITER_PREFIX):
                raise KeyboardInterrupt
            return super().generate(prompt, settings)
    with pytest.raises(KeyboardInterrupt):
        run(config, factory, llm=Crash)
    writer_calls = sum(p.startswith(WRITER_PREFIX) for p in LLM.prompts)
    run(config, factory)
    _, results, _, writes = logs(config)
    assert len(results) == 4
    assert [w.status for w in writes[:2]] == ["started", "interrupted"]
    first = next(w for w in writes if w.status == "interrupted")
    assert not any(w.status == "stored" and w.trajectory_id == first.trajectory_id for w in writes)
    assert sum(p.startswith(WRITER_PREFIX) for p in LLM.prompts) == writer_calls + 2
    # Crash after the log row but before the store insert: resume inserts from the log.
    monkeypatch.setattr(module.MemoryStore, "add", crash)
    config.run_id = "second"
    config.memory = memory_config(config.output_dir.parent / "second")
    with pytest.raises(KeyboardInterrupt):
        run(config, factory)
    monkeypatch.setattr(module.MemoryStore, "add", original)
    run(config, factory)
    _, _, _, writes = logs(config)
    assert [e.memory for e in store(config).episodes()] == [w.memory for w in writes if w.status == "stored"]


def test_retrieval_failure_aborts_task_without_fallback(experiment):
    config, _, factory = experiment
    class Broken(Embedder):
        def embed(self, texts, config):
            if texts[0].startswith("put some") and Embedder.calls:
                raise RuntimeError("secret must not persist")
            return super().embed(texts, config)
    metrics = run(config, factory, embedder_factory=Broken)
    trajectories, results, _, _ = logs(config)
    assert metrics["runtime_failures"] >= 1
    assert all("secret" not in t.model_dump_json() for t in trajectories)
    aborted = [t for t in trajectories if t.termination_reason == "runtime_error"]
    assert aborted and all(not t.steps for t in aborted)


def test_pilot_config_freezes_baseline_acting_setup():
    episodic = load_baseline_config("configs/episodic_pilot.yaml")
    baseline = load_baseline_config("configs/baseline_calibration.yaml")
    assert baseline.memory is None and episodic.memory is not None
    assert episodic.agent == baseline.agent
    assert episodic.environment == baseline.environment
    assert episodic.manifest == baseline.manifest
    assert (episodic.selection_seed, episodic.data_kind) == (baseline.selection_seed, baseline.data_kind)
    assert episodic.run_id != baseline.run_id
    assert episodic.memory.embedding.model_id == "text-embedding-3-small"


def test_memory_requires_agent_v2(tmp_path):
    with pytest.raises(ValueError):
        BaselineConfig(run_id="x", manifest=tmp_path / "m", output_dir=tmp_path,
                       environment=ALFWorldConfig(backend_config=tmp_path / "b"),
                       agent=AgentConfig(llm=LLMConfig(model_id="m")),
                       memory=memory_config(tmp_path))


def test_comparison_pairs_no_memory_and_episodic_runs(experiment):
    from aec.experiments.comparison import compare_runs
    config, manifest, factory = experiment
    memory = config.memory
    config.run_id, config.memory = "none", None
    run(config, factory)
    config.run_id, config.memory = "episodic", memory
    run(config, factory)
    comparison = compare_runs(config.output_dir / "none", config.output_dir / "episodic")
    assert (comparison["previous_policy"], comparison["candidate_policy"]) == ("no_memory", "episodic")
    assert "identical acting setup" in comparison["interpretation"]
    _, results, _, _ = logs(config)
    assert [t["candidate_retrieved_memory_ids"] for t in comparison["tasks"]] == [
        r.retrieved_memory_ids for r in results]
    assert [t["task_id"] for t in comparison["tasks"]] == list(manifest.task_ids)
