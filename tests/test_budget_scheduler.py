from treetune.ingpo.budget_scheduler import FlexibleBudgetScheduler


def test_flexible_scheduler_marks_queue_ids_and_floor_allocates():
    nodes = [
        {"ingpo_segment_id": "a", "ingpo_reward_variance": 0.0},
        {"ingpo_segment_id": "b", "ingpo_reward_variance": 0.5},
        {"ingpo_segment_id": "c", "ingpo_reward_variance": 1.0},
    ]
    scheduler = FlexibleBudgetScheduler(queue_count=2, lambda_=0.02)
    summaries = scheduler.allocate(nodes, total_depth_budget=9)

    assert summaries
    assert all("ingpo_budget_queue_id" in node for node in nodes)
    assert sum(summary.allocated_budget for summary in summaries) <= 9
    assert sum(summary.underallocated_budget for summary in summaries) == 9 - sum(
        summary.allocated_budget for summary in summaries
    )
