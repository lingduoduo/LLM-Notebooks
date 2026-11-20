"""Optional browser acceptance checks. Requires Playwright and a running dashboard.

python visuals/check_workflow_dashboard.py --url http://127.0.0.1:8765 \
  --browser '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
"""
import argparse
from pathlib import Path


def check(url, executable):
    from playwright.sync_api import sync_playwright, expect
    out = Path(__file__).resolve().parents[1] / 'out' / 'workflow'
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path': executable} if executable else {}))
        page = browser.new_page(viewport={'width': 1600, 'height': 1040})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.get_by_role('tab', name='6 Sandbox').click()
        page.get_by_role('button', name='Reset sandbox', exact=True).click()
        expect(page.get_by_role('status')).to_contain_text('restored')
        page.get_by_role('tab', name='4 Infer').click()
        expect(page.locator('#inspector')).to_contain_text('sexy latex')
        assert page.locator('.graph-node').count() == 8
        edge = page.locator('.graph-edge[data-edge-type="INFERRED_AS"]').first
        edge.focus(); edge.press('Enter')
        expect(page.locator('#inspector')).to_contain_text('Inference explanation')
        expect(page.locator('#inspector')).to_contain_text('sexy latex')
        page.screenshot(path=str(out / 'dashboard-inference-evidence.png'))

        page.get_by_role('button', name='Run inference', exact=True).click()
        expect(page.locator('#inspector')).to_contain_text('No contradictions')
        page.locator('[data-rule="material_interest"]').click()
        expect(page.locator('#inspector .derivation')).to_have_count(2)
        expect(page.locator('.graph-node.accent')).to_have_count(2)
        page.get_by_role('button', name='Clear inference', exact=True).click()

        page.get_by_role('tab', name='1 Build').click()
        expect(page.locator('.graph-node[data-node-id="type:Room"]')).to_have_count(1)
        page.locator('[data-type="Concept"]').click()
        expect(page.locator('#inspector')).to_contain_text('83 instances')
        page.get_by_label('Search concepts').fill('latex')
        expect(page.locator('#concept-list .choice')).to_have_count(1)
        page.locator('#concept-list .choice').click()
        expect(page.locator('#inspector')).to_contain_text('Material Interests')
        expect(page.locator('#inspector')).to_contain_text('Vocabulary')
        page.get_by_role('button', name='Show all 83 concepts').click()
        expect(page.locator('.graph-node')).to_have_count(83)

        page.get_by_role('tab', name='2 Map').click()
        page.locator('#controls [data-node-id="room:clear"]').click()
        expect(page.locator('.graph-node')).to_have_count(5)
        expect(page.locator('#mapping-preview')).to_contain_text('title → subject')
        expect(page.locator('#inspector')).to_contain_text('Private show with cosplay and latex')
        page.locator('.graph-node[data-node-id="source:subject"]').focus()
        page.locator('.graph-node[data-node-id="source:subject"]').press('Enter')
        expect(page.locator('#inspector')).to_contain_text('Source field')
        expect(page.locator('.graph-node.selected')).to_have_attribute('data-node-id', 'source:subject')
        page.locator('summary').click()
        page.locator('#csv-editor').fill('id,title,tags,declared\nfresh,latex party,[],[]\n')
        page.get_by_role('button', name='Preview mapping', exact=True).click()
        expect(page.locator('#inspector')).to_contain_text('Mapping preview')
        expect(page.locator('.graph-node.accent')).to_have_count(1)
        assert page.locator('.graph-node[data-node-id="room:clear"]').count() == 0, 'preview must not show workspace rooms'

        page.get_by_role('tab', name='3 Query').click()
        page.get_by_label('Operation', exact=True).select_option('search')
        page.get_by_label('Search text').fill('no-such-entity-123')
        page.get_by_role('button', name='Run query', exact=True).click()
        expect(page.locator('#graph-message')).to_be_visible()
        expect(page.locator('#inspector')).to_contain_text('No results')
        page.get_by_role('button', name='Load path example').click()
        page.get_by_role('button', name='Run query', exact=True).click()
        expect(page.locator('#inspector')).to_contain_text('1 result')
        expect(page.locator('.graph-node:not(.dim)')).to_have_count(4)
        page.get_by_role('button', name='Load pattern example').click()
        page.get_by_role('button', name='Run query', exact=True).click()
        expect(page.locator('#inspector')).to_contain_text('4 results')
        expect(page.locator('#inspector')).to_contain_text('p = Roleplay')
        page.get_by_label('Operation', exact=True).select_option('bfs')
        page.get_by_label('Starting node ID').fill('room:ambiguous')
        page.get_by_label('Maximum hops').fill('1')
        page.get_by_label('Relationship', exact=True).select_option('')
        page.get_by_role('button', name='Run query', exact=True).click()
        expect(page.locator('#inspector')).to_contain_text('8 results')

        page.get_by_role('tab', name='5 Behaviors').click()
        page.locator('[data-behavior="event:ambiguous"]').click()
        expect(page.locator('#behavior-flow')).to_contain_text('requires_context_review = true')
        expect(page.locator('#behavior-flow')).to_contain_text('create_review_task')
        page.get_by_role('button', name='Open in sandbox').click()
        page.get_by_label('Room', exact=True).select_option('ambiguous')
        # Hold a real behavior response after the server executes it. Reset must
        # stay unavailable until that response has been applied to the canvas.
        held = []
        def hold_behavior(route):
            response = route.fetch()
            held.append((route, response))
        page.route('**/api/behavior', hold_behavior)
        page.get_by_role('button', name='Run event', exact=True).click()
        expect(page.get_by_role('button', name='Reset sandbox', exact=True)).to_be_disabled()
        # Give the route callback the browser event loop until the response exists.
        page.wait_for_function('document.getElementById("workspace").getAttribute("aria-busy") === "true"')
        assert held, 'Behavior request was not intercepted'
        held[0][0].fulfill(response=held[0][1])
        page.unroute('**/api/behavior', hold_behavior)
        expect(page.locator('#inspector')).to_contain_text('review:ambiguous:context')
        expect(page.locator('.graph-node.accent[data-node-id="review:ambiguous:context"]')).to_have_count(1)
        expect(page.locator('#inspector .console')).to_contain_text('create_review_task')
        page.screenshot(path=str(out / 'dashboard-sandbox.png'))
        page.get_by_role('button', name='Run event', exact=True).click()
        expect(page.locator('#inspector')).to_contain_text('No duplicate was created')
        page.locator('[data-behavior="function:pending"]').click()
        page.get_by_role('button', name='Run function', exact=True).click()
        expect(page.locator('#inspector')).to_contain_text('review:ambiguous:context')
        expect(page.locator('#inspector')).to_contain_text('0 nodes added')
        page.get_by_role('button', name='Reset sandbox', exact=True).click()
        expect(page.get_by_role('status')).to_contain_text('restored')
        expect(page.locator('.graph-node[data-node-id="review:ambiguous:context"]')).to_have_count(0)
        page.locator('[data-behavior="action:classify"]').click()
        page.get_by_role('button', name='Run action', exact=True).click()
        expect(page.locator('#inspector .trace-step')).to_contain_text('classify_room')
        expect(page.locator('#inspector')).to_contain_text('0 nodes added')
        page.get_by_role('button', name='Reset sandbox', exact=True).click()
        expect(page.get_by_role('status')).to_contain_text('restored')
        page.get_by_role('tab', name='4 Infer').click()
        page.screenshot(path=str(out / 'dashboard-desktop.png'))

        phone = browser.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        phone.on('pageerror', lambda error: errors.append(str(error)))
        phone.goto(url)
        expect(phone.locator('#inspector')).to_contain_text('sexy latex')
        for tab in ('1 Build', '2 Map', '3 Query', '4 Infer', '5 Behaviors', '6 Sandbox'):
            phone.get_by_role('tab', name=tab).click()
            assert phone.evaluate('document.documentElement.scrollWidth <= innerWidth'), tab
        phone.get_by_role('tab', name='4 Infer').click()
        phone.locator('#graph-area').scroll_into_view_if_needed()
        assert phone.locator('.graph-node').first.bounding_box()['width'] >= 100, 'Phone graph labels are too small'
        phone.screenshot(path=str(out / 'dashboard-phone.png'))
        assert not errors, errors
        browser.close()
    print('PASS: six tabs, schema, rule inference, mapping preview, search/paths/BFS/patterns, behavior flow, sandbox event/idempotency/function/reset/log; desktop and phone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--browser', help='Existing Chromium/Chrome executable path')
    args = parser.parse_args()
    check(args.url, args.browser)
