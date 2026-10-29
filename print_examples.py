from generator import SessionConfig, generate_session

configs = [
    SessionConfig(num_initial_rules=2, num_updates=1, gap_length=3, task_id="filter_even_squares", seed=101),
    SessionConfig(num_initial_rules=3, num_updates=2, gap_length=2, task_id="most_frequent_char", seed=202),
    SessionConfig(num_initial_rules=2, num_updates=0, gap_length=2, task_id="compute_gcd", seed=303),
]

for i, cfg in enumerate(configs, 1):
    pri, zg, inl = generate_session(cfg)
    print("=" * 80)
    print(f"EXAMPLE SESSION #{i} (Task: {cfg.task_id}, Init Rules: {cfg.num_initial_rules}, Updates: {cfg.num_updates}, Gap: {cfg.gap_length})")
    print("=" * 80)
    print(f"ACTIVE RULES:     {pri.active_rules}")
    print(f"SUPERSEDED RULES: {pri.superseded_rules}")
    print(f"\n--- PRIMARY SESSION (Total Messages: {len(pri.messages)}) ---")
    for m_idx, msg in enumerate(pri.messages, 1):
        role = msg['role'].upper()
        content = msg['content']
        print(f"[{m_idx}] {role}:\n{content}\n")

    print(f"--- ZERO-GAP VARIANT (Total Messages: {len(zg.messages)}) ---")
    print("Zero-gap contains the exact same rule and update turns, with distractor gap eliminated (D=0).\n")

    print("--- INLINE CONTROL VARIANT FINAL TURN ---")
    print("Inline variant contains the exact same history, but the final prompt explicitly reminds the model:")
    print(f"{inl.messages[-1]['content']}\n\n")
