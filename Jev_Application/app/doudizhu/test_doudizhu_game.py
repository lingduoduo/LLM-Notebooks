"""Offline tests for the Dou Dizhu engine and server (local judges only)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import doudizhu_game as D
import serve_doudizhu


def test_prompt_seat_roles_follow_landlord_seed_1():
    state = D.make_doudizhu(seed=1)
    assert state['landlord'] == 1 and state['turn'] == 1
    text = D.render_request(state)['state']
    assert 'you are seat 1 (landlord)' in text
    assert 'seat 2 (peasant) has 17 cards' in text
    assert 'seat 0 (peasant) has 17 cards' in text
    assert 'peasant_a' not in text and 'peasant_b' not in text
    lead = D.gen_moves(state['hands'][1])[0]
    state = D.step(state, lead)
    text = D.render_request(state)['state']
    assert 'you are seat 2 (peasant)' in text
    assert 'Table: seat 1 (landlord) played' in text


def test_prompt_roles_for_every_landlord_seat():
    for seed in (0, 1, 2):
        state = D.make_doudizhu(seed=seed)
        text = D.render_request(state)['state']
        assert f"you are seat {state['landlord']} (landlord)" in text
        assert text.count('(landlord)') == 1


def test_step_ignores_forged_rank_and_length():
    state = D.make_doudizhu(seed=1)
    legal = next(m for m in D.gen_moves(state['hands'][1]) if m['type'] == 'single')
    forged = dict(legal, rank=99, length=7)
    after = D.step(state, forged)
    assert after['trick']['rank'] == legal['rank'] and after['trick']['length'] == 1
    assert after['history'][-1]['rank'] == legal['rank']


def test_step_rejects_cards_not_in_hand():
    state = D.make_doudizhu(seed=1)
    missing = next(c for c in range(54) if c not in state['hands'][1])
    with pytest.raises(ValueError):
        D.step(state, {'type': 'single', 'cards': [missing], 'rank': 0, 'length': 1})


def test_judge_exception_falls_back_and_thread_survives():
    def broken(request):
        raise RuntimeError('boom')

    session = serve_doudizhu.DoudizhuSession(broken, pace=0, seed=4)
    plays = session.state['plays']
    original_step = D.step

    def stop_after(state, move):
        session.alive = False
        return original_step(state, move)

    D.step = stop_after
    try:
        session.run()  # must return normally, not raise
    finally:
        D.step = original_step
    assert session.last_answer == 'm0'
    assert session.state['plays'] == plays + 1


def test_session_episodes_use_cli_seed():
    session = serve_doudizhu.DoudizhuSession(serve_doudizhu.local_judge, pace=0, seed=10)
    assert session.state == D.make_doudizhu(seed=10)
    session.state = dict(session.state, done=True, outcome='landlord_win')
    import time
    original_sleep = time.sleep

    def fast_sleep(_):
        if session.episode >= 1:
            session.alive = False

    serve_doudizhu.time.sleep = fast_sleep
    try:
        session.run()
    finally:
        serve_doudizhu.time.sleep = original_sleep
    assert session.episode == 1
    assert session.state['seed'] == 11
