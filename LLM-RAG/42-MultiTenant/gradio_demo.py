#!/usr/bin/env python3

import gradio as gr
from typing import List, Tuple, Optional, Dict, Any

from backend.compat import AIMessage, HumanMessage
from langgraph_demo import (
    MultiTenantCustomerService,
    TenantContext,
    tenant_context,
    global_session_manager,
    global_storage,
)

ChatHistory = List[Dict[str, Any]]  # {"role": "...", "content": "..."}


class GradioMultiTenantDemo:
    def __init__(self):
        self.customer_service = MultiTenantCustomerService()

        # Theme/CSS: different Gradio versions accept these in different places.
        self._ui_theme = gr.themes.Soft() if hasattr(gr, "themes") else None
        self._ui_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px;
        }
        """

        _users = [
            ("company-a", "alice", "Alice (Company A)"),
            ("company-a", "bob", "Bob (Company A)"),
            ("company-b", "charlie", "Charlie (Company B)"),
            ("company-b", "diana", "Diana (Company B)"),
            ("enterprise-x", "manager1", "Manager1 (Enterprise X)"),
            ("enterprise-x", "manager2", "Manager2 (Enterprise X)"),
        ]
        self._user_choices: List[str] = [d for _, _, d in _users]
        self._display_to_user: Dict[str, Tuple[str, str]] = {d: (t, u) for t, u, d in _users}

    def get_user_choices(self) -> List[str]:
        return self._user_choices

    def parse_user_selection(self, user_display: str) -> Tuple[str, str]:
        return self._display_to_user.get(user_display, ("company-a", "alice"))

    def get_session_info(self, tenant_id: str, user_id: str) -> str:
        session_id = global_session_manager.get_session_id(tenant_id, user_id)
        return f"Session ID: {session_id}"

    def get_isolation_status(self) -> str:
        all_sessions = global_session_manager.get_all_sessions()
        status_lines = ["**Multi-tenant Isolation Status**", "---"]

        for user_key, session_id in all_sessions.items():
            tenant_id, user_id = user_key.split("_", 1)
            context = TenantContext(tenant_id, user_id, session_id)
            state = global_storage.get_or_create_state(context)
            msg_count = len(state["messages"])
            status_lines.append(
                f"• **{tenant_id}-{user_id}**: Session `{session_id}` | Messages: {msg_count}"
            )

        if not all_sessions:
            status_lines.append("No active sessions")

        return "\n".join(status_lines)

    def process_message(
        self,
        message: str,
        user_selection: str,
        history: Optional[ChatHistory],
    ) -> Tuple[ChatHistory, str, str]:
        """Return Chatbot history in Gradio messages format."""
        if history is None:
            history = []

        if not message.strip():
            return history, "", self.get_isolation_status()

        tenant_id, user_id = self.parse_user_selection(user_selection)
        session_id = global_session_manager.get_session_id(tenant_id, user_id)

        # Add user message
        history.append({"role": "user", "content": message})

        try:
            with tenant_context(tenant_id, user_id, session_id):
                response = self.customer_service.process_message(message)

            # Add assistant message
            history.append({"role": "assistant", "content": response})

        except Exception as e:
            history.append({"role": "assistant", "content": f"Error while processing message: {str(e)}"})

        return history, "", self.get_isolation_status()

    def clear_chat(self, user_selection: str) -> Tuple[ChatHistory, str, str]:
        tenant_id, user_id = self.parse_user_selection(user_selection)

        new_session_id = global_session_manager.clear_session(tenant_id, user_id)

        return (
            [],
            f"Cleared chat history for {user_selection}\nNew Session ID: {new_session_id}",
            self.get_isolation_status(),
        )

    def switch_user(self, user_selection: str) -> Tuple[ChatHistory, str, str]:
        tenant_id, user_id = self.parse_user_selection(user_selection)
        session_id = global_session_manager.get_session_id(tenant_id, user_id)

        context = TenantContext(tenant_id, user_id, session_id)
        state = global_storage.get_or_create_state(context)
        messages = state["messages"]

        # Convert LangGraph messages -> Gradio messages format
        history: ChatHistory = []
        pending_user: Optional[str] = None

        for msg in messages:
            if isinstance(msg, HumanMessage):
                # flush previous user if it never got a reply
                if pending_user is not None:
                    history.append({"role": "user", "content": pending_user})
                    history.append({"role": "assistant", "content": "(Waiting for reply)"})
                pending_user = msg.content

            elif isinstance(msg, AIMessage):
                if pending_user is None:
                    # system-ish message
                    history.append({"role": "assistant", "content": msg.content})
                else:
                    history.append({"role": "user", "content": pending_user})
                    history.append({"role": "assistant", "content": msg.content})
                    pending_user = None

        if pending_user is not None:
            history.append({"role": "user", "content": pending_user})
            history.append({"role": "assistant", "content": "(Waiting for reply)"})

        info_msg = (
            f"Switched to {user_selection}\n"
            f"Session ID: {session_id}\n"
            f"Stored messages: {len(messages)}\n"
            f"Rendered messages: {len(history)}"
        )

        return history, info_msg, self.get_isolation_status()

    def create_interface(self):
        with gr.Blocks(title="Multi-tenant LangGraph Customer Support Demo") as demo:
            gr.HTML(
                """
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                    <h1>Multi-tenant LangGraph Customer Support Demo</h1>
                    <p>Tenant Isolation | Context Memory | Real-time AI Chat | Cross-terminal Session Consistency</p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    user_dropdown = gr.Dropdown(
                        choices=self.get_user_choices(),
                        value=self.get_user_choices()[0],
                        label="Select user identity",
                        info="Different users have independent chat spaces and memory",
                    )

                    chatbot = gr.Chatbot(
                        label="AI Customer Support Chat",
                        height=400,
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your message...",
                            label="Message input",
                            scale=4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("Clear chat", variant="secondary")
                        switch_btn = gr.Button("Switch user", variant="secondary")

                with gr.Column(scale=1):
                    gr.HTML("<h3>System Status</h3>")

                    session_info = gr.Textbox(
                        label="Session info",
                        value="Please select a user identity",
                        interactive=False,
                        lines=2,
                    )

                    isolation_status = gr.Markdown(
                        value=self.get_isolation_status(),
                        label="Isolation status",
                    )

                    refresh_btn = gr.Button("Refresh status", size="sm")

            send_btn.click(
                self.process_message,
                inputs=[msg_input, user_dropdown, chatbot],
                outputs=[chatbot, msg_input, isolation_status],
            )

            msg_input.submit(
                self.process_message,
                inputs=[msg_input, user_dropdown, chatbot],
                outputs=[chatbot, msg_input, isolation_status],
            )

            clear_btn.click(
                self.clear_chat,
                inputs=[user_dropdown],
                outputs=[chatbot, session_info, isolation_status],
            )

            switch_btn.click(
                self.switch_user,
                inputs=[user_dropdown],
                outputs=[chatbot, session_info, isolation_status],
            )

            user_dropdown.change(
                self.switch_user,
                inputs=[user_dropdown],
                outputs=[chatbot, session_info, isolation_status],
            )

            refresh_btn.click(
                self.get_isolation_status,
                outputs=[isolation_status],
            )

            demo.load(
                lambda: (
                    self.get_session_info("company-a", "alice"),
                    self.get_isolation_status(),
                ),
                outputs=[session_info, isolation_status],
            )

        return demo


def main():
    print("Starting multi-tenant LangGraph customer support demo...")

    demo_app = GradioMultiTenantDemo()
    demo = demo_app.create_interface()

    print("System started successfully!")

    launch_kwargs = dict(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
    )

    # Apply theme/css if supported; otherwise fall back cleanly
    try:
        if demo_app._ui_theme is not None:
            launch_kwargs["theme"] = demo_app._ui_theme
        launch_kwargs["css"] = demo_app._ui_css
        demo.launch(**launch_kwargs)
    except TypeError:
        launch_kwargs.pop("theme", None)
        launch_kwargs.pop("css", None)
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
