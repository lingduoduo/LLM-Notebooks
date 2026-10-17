"""Offline tests for the Sudoku runner (no API calls)."""
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Different apps ship a module named experiment; load this app explicitly.
_spec = importlib.util.spec_from_file_location("sudoku_experiment", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)
import jev_sudoku
import sudoku_game as S


def test_local_judge_episode_logs_int_gold_and_correct_flags():
    result = jev_sudoku.run_episode(1550, 55, jev_sudoku.local_judge, trial_memory=True)
    assert result['outcome'] == 'win'
    steps = result['steps_detail']
    assert steps and all(type(s['gold']) is int for s in steps)
    assert all(s['correct'] for s in steps)
    assert all(s['correct'] == (s['answered'] == s['gold']) for s in steps)


def test_trial_memory_never_records_correct_digit_as_wrong():
    seen_wrong = []

    def spy(request):
        seen_wrong.append(request.get('wrong_digits', {}))
        return jev_sudoku.local_judge(request)

    result = jev_sudoku.run_episode(1550, 55, spy, trial_memory=True)
    gold = {tuple(s['cell']): s['gold'] for s in result['steps_detail'] if s['correct']}
    for wrong in seen_wrong:
        for key, digits in wrong.items():
            r, c = map(int, key.split(','))
            assert gold.get((r, c)) not in digits


def test_experiment_uses_shipped_local_judge():
    assert experiment.local_judge is jev_sudoku.local_judge
    assert not hasattr(experiment, 'judges')


def test_make_sudoku_reports_actual_hole_count():
    state = S.make_sudoku(seed=10, holes=55)
    actual = sum(1 for row in state['grid'] for v in row if v == 0)
    assert state['holes'] == actual < 55  # seed 10 只能挖出 54 个
    S.validate_state(state)
    result = jev_sudoku.run_episode(10, 55, jev_sudoku.local_judge)
    assert result['holes'] == actual and result['holes_requested'] == 55


def test_render_request_constraint_text_capitalised_once():
    state = S.make_sudoku(seed=1, holes=40)
    text = S.render_request(state)['state']
    assert '. Row ' in text


def test_zero_hole_puzzle_is_already_won():
    state = S.make_sudoku(seed=4, holes=0)
    S.validate_state(state)
    assert state['done'] and state['outcome'] == 'win'
    result = jev_sudoku.run_episode(4, 0, jev_sudoku.local_judge)
    assert result['outcome'] == 'win' and result['steps'] == 0


def test_nested_generation_does_not_change_seeded_puzzle(monkeypatch):
    # Re-enter generation during a candidate calculation, as concurrent callers can.
    expected = S.make_sudoku(seed=7, holes=40)
    original = S._candidates
    entered = False

    def candidates(grid, row, col):
        nonlocal entered
        if not entered:
            entered = True
            S.make_sudoku(seed=99, holes=1)
        return original(grid, row, col)

    monkeypatch.setattr(S, '_candidates', candidates)
    assert S.make_sudoku(seed=7, holes=40) == expected


def test_validation_checks_boxes_outside_first_band():
    import pytest

    state = S.make_sudoku(seed=4, holes=0)
    # Swapping rows across the lower bands keeps every row and column valid,
    # but can repeat digits in lower boxes while the top-left box stays valid.
    for key in ('grid', 'solution'):
        state[key][3], state[key][6] = state[key][6], state[key][3]
    with pytest.raises(ValueError, match='row, column, or box'):
        S.validate_state(state)
