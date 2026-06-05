import sys
from core.llm import LLMEngine
from core.memory import MemoryStore

def main():
    print("Initializing AI Engines...")
    try:
        llm = LLMEngine()
        memory = MemoryStore()
    except Exception as e:
        print(f"Failed to initialize engines: {e}")
        sys.exit(1)
        
    session_stats = {
        'last_query': None,
        'last_emb_sec': 0.0,
        'last_ttft': 0.0,
        'last_tok_sec': 0.0
    }

    print("\n" + "="*50)
    print(" Edge AI Assistant (Terminal Mode)")
    print(" Type 'exit' or 'quit' to close.")
    print(" Type '/help' for a list of interactive commands.")
    print("="*50 + "\n")

    import os
    import time
    import psutil

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            # Command Router
            if user_input.startswith("/"):
                cmd_parts = user_input.split(" ", 1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
                
                if cmd in ['/exit', '/quit']:
                    print("Goodbye!")
                    break
                elif cmd == '/help':
                    print("\nAvailable Commands:")
                    print("  /help                     - Show this help message")
                    print("  /stats                    - Show real-time performance and memory statistics")
                    print("  /memories                 - List all stored memories in SQLite")
                    print("  /delete <id>              - Delete a specific memory by ID")
                    print("  /clear                    - Clear all memories from the database")
                    print("  /system [new prompt]      - View or update the LLM system prompt")
                    print("  /exit or /quit            - Exit the application")
                elif cmd == '/stats':
                    process = psutil.Process(os.getpid())
                    ram_mb = process.memory_info().rss / (1024 * 1024)
                    print("\n" + "="*50)
                    print("             SYSTEM & MODEL STATS")
                    print("="*50)
                    print(f"RAM Footprint : {ram_mb:.2f} MB")
                    print(f"SQLite DB File: {memory.count_memories()} memories stored")
                    print(f"LLM Model     : {llm.model_path}")
                    print(f"Metal GPU Accel: {'Enabled' if llm.n_gpu_layers != 0 else 'Disabled'} (n_gpu_layers={llm.n_gpu_layers})")
                    print(f"Context Window: {llm.ctx_size} tokens")
                    print("-"*50)
                    print("             LAST INTERACTION METRICS")
                    print("-"*50)
                    if session_stats['last_query'] is not None:
                        print(f"Prompt        : '{session_stats['last_query']}'")
                        print(f"Embedding Time: {session_stats['last_emb_sec'] * 1000:.1f} ms")
                        print(f"Time-to-First-Token (TTFT): {session_stats['last_ttft']:.3f} s")
                        print(f"Throughput    : {session_stats['last_tok_sec']:.1f} tokens/sec")
                    else:
                        print("No interactions recorded in this session yet.")
                    print("="*50 + "\n")
                elif cmd == '/memories':
                    mems = memory.get_all()
                    if not mems:
                        print("Memory store is empty.")
                    else:
                        print(f"\n--- Stored Memories ({len(mems)}) ---")
                        for m in mems:
                            ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m['timestamp']))
                            text_snippet = m['text'].replace('\n', ' ')
                            if len(text_snippet) > 80:
                                text_snippet = text_snippet[:77] + "..."
                            print(f"[{m['id']}] [{ts_str}] ({m['source']}): {text_snippet}")
                elif cmd == '/delete':
                    if not arg:
                        print("Error: Please provide a memory ID to delete. Usage: /delete <id>")
                    else:
                        try:
                            mem_id = int(arg)
                            all_mems = memory.get_all()
                            exists = any(m['id'] == mem_id for m in all_mems)
                            if not exists:
                                print(f"Memory with ID {mem_id} not found.")
                            else:
                                memory.delete(mem_id)
                                print(f"Successfully deleted memory ID {mem_id}.")
                        except ValueError:
                            print("Error: Memory ID must be an integer.")
                elif cmd == '/clear':
                    confirm = input("Are you sure you want to delete all memories? (y/N): ").strip().lower()
                    if confirm == 'y':
                        memory.clear_all()
                        print("All memories deleted.")
                    else:
                        print("Aborted.")
                elif cmd == '/system':
                    if not arg:
                        print(f"\nCurrent System Prompt:\n{llm.system_prompt}")
                    else:
                        llm.system_prompt = arg
                        print(f"System prompt updated to:\n{llm.system_prompt}")
                else:
                    print(f"Unknown command: {cmd}. Type /help for a list of commands.")
                continue
                
            # Search memory context (measure embedding latency)
            t_emb_start = time.time()
            relevant_mems = memory.search(user_input, top_k=3)
            context_str = memory.format_context(relevant_mems)
            emb_sec = time.time() - t_emb_start
            
            print("Assistant: ", end="", flush=True)
            full_response = ""
            ttft = 0.0
            tok_sec = 0.0
            
            for output in llm.generate(user_input, context=context_str):
                token = output["token"]
                full_response += token
                print(token, end="", flush=True)
                ttft = output["metrics"]["ttft"]
                tok_sec = output["metrics"]["tok_sec"]
            print() # Newline after response completes
            
            # Save interaction to memory
            memory.add(f"User: {user_input}\nAssistant: {full_response}")
            
            # Update session stats
            session_stats['last_query'] = user_input
            session_stats['last_emb_sec'] = emb_sec
            session_stats['last_ttft'] = ttft
            session_stats['last_tok_sec'] = tok_sec
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
