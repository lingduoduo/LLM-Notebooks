#!/usr/bin/env python3
"""
Compatibility wrapper around the layered multi-tenant backend.

This keeps the original public surface for the Gradio demo while moving the
actual implementation into dedicated access/business/execution/storage/model
modules under `backend/`.
"""

from __future__ import annotations

from typing import List, Optional

from backend import (
    MultiTenantCustomerService,
    TenantContext,
    get_current_context,
    global_platform_service,
    global_session_manager,
    global_storage,
    tenant_context,
)


def list_user_sessions(tenant_id: str, user_id: str) -> List[str]:
    return global_storage.list_sessions(tenant_id, user_id)


def interactive_session(tenant_id: str, user_id: str, session_id: Optional[str] = None) -> None:
    service = MultiTenantCustomerService()
    resolved_session_id = session_id or global_session_manager.get_session_id(tenant_id, user_id)
    print(f"[session] tenant:{tenant_id} | user:{user_id} | session_id:{resolved_session_id}")
    print("Type 'exit' to quit\n")

    with tenant_context(tenant_id, user_id, resolved_session_id):
        while True:
            user_input = input("User >>> ")
            if user_input.lower() == "exit":
                break
            answer = service.process_message(user_input)
            print(f"\nAI >>> {answer}")


def main() -> None:
    print("=== Multi-tenant Customer Support Demo ===")
    print("Backend layers: FastAPI gateway -> Dify-style service -> LangGraph -> storage/model")

    while True:
        print("\nChoose an action:")
        print("1. Start chat")
        print("2. View session list")
        print("3. Show isolation status")
        print("4. Exit")

        choice = input("Select (1-4): ").strip()
        if choice == "1":
            tenant_id = input("Enter tenant_id: ").strip()
            user_id = input("Enter user_id: ").strip()
            interactive_session(tenant_id, user_id)
        elif choice == "2":
            tenant_id = input("Enter tenant_id: ").strip()
            user_id = input("Enter user_id: ").strip()
            print(list_user_sessions(tenant_id, user_id))
        elif choice == "3":
            print(global_storage.show_isolation_status())
        elif choice == "4":
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
