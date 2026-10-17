"""Offline tests for the Blackjack engine, experiment and server (local judges only)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blackjack_game as B
import experiment
import serve_blackjack

# card id = suit * 13 + rank index (0 = A, 5 = '6', 4 = '5', 9 = '10', 12 = 'K')
ACE, FIVE, SIX, TEN, KING = 0, 4, 5, 9, 12


def _player_state(player, dealer, shoe, bank=100):
    return {'player': player, 'dealer': dealer, 'shoe': shoe, 'shoe_index': 0,
            'phase': 'player', 'bank': bank, 'bet': B.BET}


def test_soft_totals():
    assert B.hand_value([ACE, SIX]) == (17, True)
    assert B.hand_value([ACE, SIX, KING]) == (17, False)
    assert B.hand_value([ACE, 13 + ACE]) == (12, True)
    assert B.hand_value([KING, SIX]) == (16, False)


def test_basic_strategy_soft_17_vs_5_doubles():
    state = _player_state([ACE, SIX], [13 + KING, FIVE], shoe=list(range(52)))
    assert B.basic_strategy(state) == 'double'
    assert B.basic_strategy(dict(state, bank=B.BET)) == 'hit'  # cannot afford the double


def test_bust_probability_examples():
    aces = [13, 26, 39]
    # hard 15 vs an all-aces unseen set (shoe + hole ace): an ace never busts
    assert B.bust_probability(_player_state([TEN, FIVE], [ACE, SIX], aces)) == 0.0
    tens = [13 + TEN, 26 + TEN, 39 + TEN]
    assert B.bust_probability(_player_state([KING, FIVE], [TEN, SIX], tens)) == 1.0
    # soft 17 cannot bust on one card
    assert B.bust_probability(_player_state([ACE, SIX], [TEN, FIVE], tens)) == 0.0


def test_published_composition_is_unseen_multiset_including_hole():
    state = B.make_blackjack(seed=0)
    for _ in range(300):
        if state['done']:
            break
        assert state['phase'] == 'player'
        hole = state['dealer'][0]
        visible = set(state['shoe'][:state['shoe_index']]) - {hole}
        expected = Counter(B.RANKS[c % 13] for c in range(52) if c not in visible)
        text = B.render_request(state)['state']
        published = json.loads(text.split('Unseen cards (shoe + dealer hole card): ')[1]
                               .split('. Probability')[0])
        assert published == dict(expected)
        unseen = [c for c in range(52) if c not in visible]
        busts = sum(1 for c in unseen if B.hand_value(state['player'] + [c])[0] > 21)
        assert B.bust_probability(state) == round(busts / len(unseen), 3)
        state = B.step(state, B.basic_strategy(state))


def test_no_duplicate_cards_over_long_run():
    actions = ['hit', 'stand', 'hit', 'hit', 'double', 'stand', 'hit']
    for seed in (0, 1, 7):
        state = B.make_blackjack(seed=seed)
        history, step = [], 0
        while len(history) < 3000:
            state['bank'] = 10 ** 9  # keep the run going
            history.extend(state['history'])
            state['history'] = []  # keep per-step deepcopies cheap
            assert sorted(state['shoe']) == list(range(52))
            in_hand = state['player'] + state['dealer']
            assert len(set(in_hand)) == len(in_hand)
            assert set(in_hand) <= set(state['shoe'][:state['shoe_index']])
            action = actions[step % len(actions)]
            if action not in B.legal_actions(state):
                action = 'hit'
            state = B.step(state, action)
            step += 1
        for entry in history:
            cards = entry['player'] + entry['dealer']
            assert len(set(cards)) == len(cards), (seed, entry)


def test_double_natural_is_push():
    state = {'player': [ACE, KING], 'dealer': [13 + ACE, 13 + KING], 'bet': B.BET,
             'doubled': False, 'natural': True, 'dealer_natural': True, 'bank': 100,
             'round': 1, 'history': [], 'done': False}
    settled = B._settle(state)
    assert settled['outcome'] == 'push' and settled['last_delta'] == 0
    assert settled['bank'] == 100
    alone = B._settle(dict(state, dealer=[13 + TEN, 13 + KING], dealer_natural=False))
    assert alone['outcome'] == 'player_blackjack' and alone['last_delta'] == 15


def test_experiment_rounds_equal_history_length():
    for seed in (1, 2, 3):
        captured = {}
        original = B.step

        def spy(state, action):
            captured['state'] = original(state, action)
            return captured['state']

        B.step = spy
        try:
            result = experiment.play(seed, experiment.basic_judge)
        finally:
            B.step = original
        history = captured['state']['history']
        assert result['rounds'] == len(history)
        outcomes = Counter(e['outcome'] for e in history)
        for key, rate in result['rates'].items():
            assert rate == round(outcomes.get(key, 0) / len(history), 3)


def test_judges_match_basic_strategy():
    state = B.make_blackjack(seed=3)
    for _ in range(200):
        if state['done']:
            break
        request = B.render_request(state)
        gold = B.basic_strategy(state)
        assert experiment.basic_judge(request)['action'] == gold
        assert serve_blackjack.local_judge(request)['action'] == gold
        state = B.step(state, gold)


def test_state_json_masks_hole_card_during_player_phase():
    session = serve_blackjack.BlackjackSession(serve_blackjack.local_judge, pace=0, seed=1)
    assert session.state['phase'] == 'player'
    snap = session.snapshot()
    hole = B.card_name(session.state['dealer'][0])
    assert snap['hole_hidden'] is True
    assert snap['dealer'][0] == '??' and snap['dealer'][1] == B.card_name(session.state['dealer'][1])
    assert snap['dealer_value'] is None
    assert hole not in json.dumps(snap['dealer'])
    session.state = dict(session.state, phase='done_round')
    snap = session.snapshot()
    assert snap['dealer'][0] == hole and snap['dealer_value'] is not None


def test_judge_exception_falls_back_to_stand():
    def broken(request):
        raise RuntimeError('boom')

    session = serve_blackjack.BlackjackSession(broken, pace=0, seed=1)
    rounds = session.state['round']
    original_step = B.step

    def stop_after(state, action):
        session.alive = False
        assert action == 'stand'
        return original_step(state, action)

    B.step = stop_after
    try:
        session.run()  # must return normally, not raise
    finally:
        B.step = original_step
    assert session.last_answer == 'stand'
    assert session.state['round'] > rounds
