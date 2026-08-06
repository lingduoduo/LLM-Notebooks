"""A multi-turn customer-refund agent task used for cost analysis.

The experiment uses a controlled tool environment so results do not depend on
random tool selection. Each turn feeds a predefined tool result to a real LLM,
so every model call, token count, and calculated cost is genuine.

KV-cache friendliness and context compression are independent switches, which
produce a complete 2x2 comparison. The legacy ``run_naive`` and
``run_optimized`` entry points remain available.
"""

import uuid
from functools import lru_cache

from config import MODEL, Pricing
from tracer import Tracer

# Number of recent tool results retained in full; older results are summarized.
KEEP_VERBOSE = 2

# Keep output lengths comparable so generation variance does not distort the A/B test.
MAX_OUTPUT_TOKENS = 160

# ---------------------------------------------------------------------------
# A long, stable system prompt and tool manual (>1,024 tokens) enables caching.
# ---------------------------------------------------------------------------
STABLE_SYSTEM_PROMPT = """You are a senior customer-support agent for CloudShop, specializing in after-sales service and refunds. Follow every rule below.

# Role and objective
Help customers with refunds, returns, exchanges, and delivery questions efficiently, politely, and within platform policy. Clarify requests, verify order status, determine policy eligibility, and act within your authority.

# Tool manual
1. query_order(order_id): Fetch order details. Fields include order_id, status, item_name, sku, price, quantity, pay_time, pay_channel, buyer_note, seller_note, is_prepaid, warehouse, and promotion_tags.
2. query_logistics(order_id): Fetch shipment history. Fields include carrier, tracking_no, current_status, last_scan_time, last_scan_location, estimated_delivery, and full_trace.
3. check_refund_policy(sku, reason): Fetch the refund policy for a SKU and reason. Fields include refundable, need_return, restocking_fee_rate, refund_window_days, special_notes, and approval_required.
4. query_user_history(user_id): Fetch customer history for risk assessment. Fields include total_orders, refund_count_90d, dispute_count, risk_level, vip_tier, and register_days.
5. issue_refund(order_id, amount, reason): Initiate a refund. Fields include refund_id, status, expected_arrival, channel, and operator. Call it only when policy permits and the amount does not exceed the amount paid.
6. send_notification(user_id, channel, template, params): Notify the customer through SMS, app, or email.

# Decision rules
- First verify that the order exists and its status permits a refund. Unshipped, shipped-but-undelivered, and recently delivered orders follow different paths.
- Unshipped orders may receive an immediate full refund without a return.
- For shipped but undelivered orders, intercept the shipment or wait for its return; initiate the refund after receipt.
- Delivered items without defects may qualify for a seven-day change-of-mind return and a restocking fee.
- Defective items qualify for a full refund with no fee when the customer supplies evidence.
- Refunds over $500 or high-risk customers require manual approval.
- At each step, briefly explain which information supports the next tool call or conclusion.

# Response requirements
- Be professional, concise, and empathetic.
- Advance only one step per turn and never invent unavailable tool data.
- In the final resolution, state the refund amount, arrival time, and next actions.

# Compliance and risk
- Never disclose another customer's information or promise compensation beyond policy.
- Treat suspected fraud cautiously and trigger manual approval when appropriate.
Always follow all of these rules."""


# ---------------------------------------------------------------------------
# Fixed multi-turn scenario: a customer request followed by backend tool results.
# ---------------------------------------------------------------------------
USER_REQUEST = (
    "Hello. The Bluetooth headphones I bought last week (order ORD20240517001) "
    "have not connected since delivery. I tried everything and want a return and refund."
)

# Each tuple contains a logical step, tool name, and deliberately verbose result.
_LOGISTICS_TRACE = ",".join(
    '{"time":"2024-05-%02dT%02d:%02d","loc":"%s","desc":"%s","operator":"SF%04d","scan_type":"auto"}'
    % (17 + i // 6, 6 + i, (i * 7) % 60, loc, desc, 1000 + i)
    for i, (loc, desc) in enumerate([
        ("East China Warehouse 1", "Package collected; weight 0.42 kg"), ("East China Sorting Center", "Sorted for transfer to Shanghai"),
        ("Shanghai Transfer Center", "Arrived at transfer center"), ("Shanghai Transfer Center", "Departed in transit"),
        ("Suzhou Hub", "In transit through hub"), ("Shanghai Pudong Depot", "Arrived at delivery depot"),
        ("Shanghai Pudong Depot", "Scheduled for delivery"), ("Pudong Service Point", "Out for delivery; contacting recipient"),
        ("Pudong Service Point", "First delivery call unanswered"), ("Pudong Service Point", "Second delivery attempt"),
        ("Pudong Service Point", "Delivered; signed by recipient"),
    ])
)

TOOL_RESULTS = [
    ("turn-1", "query_order",
     '{"order_id":"ORD20240517001","status":"SIGNED","item_name":"Acme Active Noise-Canceling Headphones Pro",'
     '"sku":"SKU-BT-9981","price":499.00,"quantity":1,"pay_time":"2024-05-17T10:22:31",'
     '"pay_channel":"wechat_pay","buyer_note":"Please ship soon; this is a gift","seller_note":"Inventory verified",'
     '"is_prepaid":true,"warehouse":"East China Warehouse 1","promotion_tags":["30 off 300","Member Day","New Customer"],'
     '"actual_paid":469.00,"coupon_used":"CPN-30","points_earned":469,"invoice_requested":false,'
     '"sign_time":"2024-05-19T14:03:11","after_sale_window_end":"2024-05-26T23:59:59",'
     '"sub_items":[{"sku":"SKU-BT-9981","name":"Headphones","qty":1},{"sku":"SKU-BT-9981-CASE","name":"Charging case","qty":1},{"sku":"SKU-BT-9981-TIP","name":"Ear-tip set","qty":1}],'
     '"address_hash":"a1b2c3d4","channel":"app","device":"iOS","order_source":"home_page_recommendation"}'),
    ("turn-2", "query_logistics",
     '{"carrier":"SF Express","tracking_no":"SF1234567890123","current_status":"DELIVERED",'
     '"last_scan_time":"2024-05-19T14:03:11","last_scan_location":"Pudong Service Point, Shanghai",'
     '"estimated_delivery":"2024-05-19","weight_kg":0.42,"volume":"20x15x8cm","insured":true,'
     '"full_trace":[' + _LOGISTICS_TRACE + ']}'),
    ("turn-3", "check_refund_policy",
     '{"sku":"SKU-BT-9981","reason":"quality_issue_cannot_connect","refundable":true,'
     '"need_return":true,"restocking_fee_rate":0.0,"refund_window_days":7,'
     '"special_notes":"Defect refunds have no restocking fee and require a returned item plus inspection. '
     'If inspection finds customer damage or unauthorized repair, the item is returned without a refund. '
     'The platform pays return shipping; request a digital label. Refunds start within one business day after inspection. '
     'Activated or account-bound electronics must be unlinked before return.",'
     '"approval_required":false,"category":"consumer_electronics","quality_claim_supported":true,'
     '"return_label_provided":true,"qc_sla_days":2,"related_policy_ids":["P-3C-01","P-3C-07","P-QC-12"]}'),
    ("turn-4", "query_knowledge_base",
     '{"query":"Bluetooth headphones cannot connect troubleshooting","hits":['
     '{"kb_id":"KB-1001","title":"Common Bluetooth connection problems","content":"1. Pairing mode is off; '
     '2. Forget and reconnect the device; 3. Old firmware; 4. Low battery; 5. Another device owns the connection."},'
     '{"kb_id":"KB-1002","title":"Resetting the Acme Pro series","content":"Hold the charging-case button for 15 seconds until the light flashes red and white. '
     'Remove the old pairing on the phone and search again. If the device remains undiscoverable, treat it as a likely hardware defect."},'
     '{"kb_id":"KB-1003","title":"Defect assessment standard","content":"A failed reset, failure with another device, and no liquid or physical damage '
     'normally qualifies as a defect and supports a free return."}],"suggested_action":"Guide the customer through a reset; if it fails, use the defect-refund process"}'),
    ("turn-5", "query_user_history",
     '{"user_id":"U-88123","total_orders":37,"refund_count_90d":1,"dispute_count":0,'
     '"risk_level":"low","vip_tier":"gold","register_days":1180,"payment_disputes":0,'
     '"avg_order_value":312.5,"last_refund_reason":"wrong_size","chargeback_count":0,'
     '"complaint_count":0,"account_status":"normal","fraud_flags":[],"lifetime_value":11562.5}'),
    ("turn-6", "issue_refund",
     '{"refund_id":"RF20240520777","status":"APPROVED","amount":469.00,'
     '"expected_arrival":"1-3 business days","channel":"original_payment_wechat","operator":"agent-bot",'
     '"return_shipping":"platform_paid","return_address":"East China Warehouse 1 Returns","return_label":"SF-RET-998877",'
     '"qc_required":true,"qc_deadline":"2024-05-27","refund_flow":"pending_return->qc->refund"}'),
    ("turn-7", "send_notification",
     '{"user_id":"U-88123","channel":"app","template":"refund_approved",'
     '"delivered":true,"message_id":"MSG-556677","sent_time":"2024-05-20T15:20:03",'
     '"params":{"refund_id":"RF20240520777","amount":469.00,"return_label":"SF-RET-998877"},'
     '"read_receipt":false,"fallback_sms_scheduled":true}'),
    ("turn-8", "close_ticket",
     '{"ticket_id":"TK-20240520-3345","status":"resolved","resolution":"refund_after_return",'
     '"csat_survey_sent":true,"handle_time_s":184,"escalated":false,"agent":"agent-bot",'
     '"summary_logged":true,"tags":["refund","defect","electronics","closed"]}'),
]

# One-line summaries used by the context-compression strategy.
TOOL_SUMMARIES = {
    "turn-1": "[Summary] Order ORD20240517001: Acme Pro headphones, $469 paid, delivered May 19; return window ends May 26.",
    "turn-2": "[Summary] SF Express delivered the package May 19 at 14:03; all 11 tracking events are normal.",
    "turn-3": "[Summary] Defects qualify for a fee-free refund after a platform-paid return and inspection; no manual approval needed.",
    "turn-4": "[Summary] Knowledge base: reset the headphones first; a failed reset supports a free defect return.",
    "turn-5": "[Summary] Customer risk: 37 orders, one refund in 90 days, low risk, Gold member, no fraud flags.",
    "turn-6": "[Summary] Refund RF20240520777 for $469 initiated to WeChat after a platform-paid return and inspection.",
    "turn-7": "[Summary] Customer notified in the app that the refund was approved; return label included.",
}


def _next_user_msg(tool_name: str, tool_result: str) -> str:
    """Wrap a tool result as the next user message sent to the model."""
    return (
        f"[Result from tool {tool_name}]\n{tool_result}\n\n"
        "Explain your reasoning from this result and decide the next action."
    )


@lru_cache(maxsize=1)
def _encoder():
    """Return a tokenizer used to estimate tokens injected by tool results."""
    import tiktoken
    try:
        return tiktoken.encoding_for_model(MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _ntok(text: str) -> int:
    return len(_encoder().encode(text))


# ---------------------------------------------------------------------------
# Four A/B scenarios: display name plus the two optimization switches.
# ---------------------------------------------------------------------------
SCENARIOS = {
    "naive":    ("A Naive (no cache/no compression)", False, False),
    "kv":       ("KV cache only (stable prefix/no compression)", True, False),
    "compress": ("Compression only (unstable prefix/summaries)", False, True),
    "both":     ("B Optimized (KV cache + compression)", True, True),
}


def build_messages(idx, step, tool, result, turns, kv_cache, compress):
    """Build messages for turn ``idx`` and return cumulative tool-result tokens."""
    if kv_cache:
        system = {"role": "system", "content": STABLE_SYSTEM_PROMPT}
    else:
        volatile_head = f"[Session trace] session={uuid.uuid4()} request={uuid.uuid4()}\n\n"
        system = {"role": "system", "content": volatile_head + STABLE_SYSTEM_PROMPT}

    history = [{"role": "user", "content": USER_REQUEST}]
    tool_ctx_tokens = 0
    for j, (p_step, p_assistant, p_tool, p_result) in enumerate(turns):
        history.append({"role": "assistant", "content": p_assistant})
        if compress and idx - j > KEEP_VERBOSE:
            compact = TOOL_SUMMARIES.get(p_step, f"[Summary] {p_tool} completed.")
            history.append({"role": "user", "content": compact})
            tool_ctx_tokens += _ntok(compact)
        else:
            history.append({"role": "user", "content": _next_user_msg(p_tool, p_result)})
            tool_ctx_tokens += _ntok(p_result)

    messages = [system] + history + [
        {"role": "user", "content": _next_user_msg(tool, result)}
    ]
    tool_ctx_tokens += _ntok(result)   # Newly injected tool result for this turn.
    return messages, tool_ctx_tokens


def run_scenario(client, kv_cache: bool, compress: bool, name: str = None,
                 pricing: Pricing = None) -> Tracer:
    """Run one cell of the 2x2 experiment over the eight-turn refund task."""
    tracer = Tracer(client, name=name or f"kv={kv_cache},compress={compress}",
                    pricing=pricing)
    turns = []
    for idx, (step, tool, result) in enumerate(TOOL_RESULTS):
        messages, tool_ctx = build_messages(
            idx, step, tool, result, turns, kv_cache, compress)
        resp = tracer.chat(step=step, tool=tool, tool_ctx_tokens=tool_ctx,
                           model=MODEL, messages=messages, temperature=0,
                           max_tokens=MAX_OUTPUT_TOKENS)
        assistant_text = resp.choices[0].message.content or ""
        turns.append((step, assistant_text, tool, result))
    return tracer


def run_naive(client, pricing: Pricing = None) -> Tracer:
    """Naive approach: unstable prefix and uncompressed history."""
    return run_scenario(client, kv_cache=False, compress=False,
                        name=SCENARIOS["naive"][0], pricing=pricing)


def run_optimized(client, pricing: Pricing = None) -> Tracer:
    """Optimized approach: stable cacheable prefix plus compressed history."""
    return run_scenario(client, kv_cache=True, compress=True,
                        name=SCENARIOS["both"][0], pricing=pricing)
