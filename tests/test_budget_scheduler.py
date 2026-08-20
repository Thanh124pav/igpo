from treetune.ingpo.budget_scheduler import FlexibleBudgetScheduler


def test_flexible_scheduler_marks_queue_ids_without_fragmenting_budget():
    nodes = [
        {"ingpo_segment_id": "a", "ingpo_reward_variance": 0.0},
        {"ingpo_segment_id": "b", "ingpo_reward_variance": 0.5},
        {"ingpo_segment_id": "c", "ingpo_reward_variance": 1.0},
    ]
    scheduler = FlexibleBudgetScheduler(queue_count=2, lambda_=0.02, n_min=1)
    summaries = scheduler.allocate(nodes, total_depth_budget=9)

    assert len(summaries) == 1
    assert all("ingpo_budget_queue_id" in node for node in nodes)
    assert summaries[0].allocated_budget == 9
    assert summaries[0].underallocated_budget == 0


def test_flexible_scheduler_passes_n_min_and_uniform_fallback():
    nodes = [
        {"ingpo_segment_id": "a", "ingpo_reward_variance": 0.0},
        {"ingpo_segment_id": "b", "ingpo_reward_variance": 0.0},
    ]
    scheduler = FlexibleBudgetScheduler(queue_count=1, lambda_=0.02, n_min=1)

    summaries = scheduler.allocate(nodes, total_depth_budget=4)

    assert summaries[0].allocations == {"a": 2, "b": 2}
    assert summaries[0].allocated_budget == 4
