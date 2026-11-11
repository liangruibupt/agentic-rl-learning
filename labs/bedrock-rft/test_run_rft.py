"""Offline tests for run_rft.py helpers with a fake OpenAI client."""
from types import SimpleNamespace as NS

import run_rft


class FakeJobs:
    def __init__(self, events, page_size):
        self.events, self.page_size, self.calls = events, page_size, []

    def list_events(self, fine_tuning_job_id, limit, after=None):
        self.calls.append(after)
        # newest first, like the OpenAI API
        ordered = sorted(self.events, key=lambda e: -e.created_at)
        start = 0 if after is None else [e.id for e in ordered].index(after) + 1
        page = ordered[start:start + min(limit, self.page_size)]
        return NS(data=page, has_more=start + len(page) < len(ordered))


def test_all_events_paginates_and_sorts_oldest_first():
    evs = [NS(id=f"ev{i}", created_at=1000 + i, message=f"m{i}", data=None) for i in range(7)]
    jobs = FakeJobs(evs, page_size=3)
    client = NS(fine_tuning=NS(jobs=jobs))
    got = run_rft.all_events(client, "ftjob-1", page=3)
    assert [e.id for e in got] == [f"ev{i}" for i in range(7)]
    assert len(jobs.calls) == 3 and jobs.calls[0] is None


def test_metric_line_uses_documented_keys():
    data = {"total_steps": 67, "actor_grad_norm": 0.00087, "response_length_mean": 519.09, "step": 4,
            "actor_pg_loss": 0.10, "critic_rewards_mean": 0.4375, "actor_entropy": 0.62, "critic_advantages_mean": 0.014}
    line = run_rft.metric_line(data)
    assert "step 4/67" in line and "reward=0.438" in line and "len=519" in line
    assert set(run_rft.METRIC_KEYS) == set(data)
