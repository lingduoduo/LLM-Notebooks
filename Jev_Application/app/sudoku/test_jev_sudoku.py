"""Offline tests for the Sudoku runner (no API calls)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment
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
