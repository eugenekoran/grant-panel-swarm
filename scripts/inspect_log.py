"""Compact timeline of a run log: python scripts/inspect_log.py runs/<name>/events.jsonl"""
import json, collections, sys
ev=[json.loads(l) for l in open(sys.argv[1])]
print(collections.Counter(e['event'] for e in ev))
for e in ev:
    if e['event']=='llm_call':
        print(f"{e['t']:7.1f} {e['agent']} t{e['turn']} in={e['prompt_tokens']} out={e['completion_tokens']} {e['finish_reason']} {e.get('provider')} {[t['name'] for t in e['tool_calls']]} reas={len(e['reasoning'] or '')} {(e['content'] or '')[:80]!r}")
    elif e['event'] in ('turn_error','invalid_reply','state_change','compaction_end','nudge','abstention','terminate','report_filed','agent_finished'):
        print('  >>', e['event'], {k:(str(v)[:160]) for k,v in e.items() if k not in('ts','snapshot','event')})
    elif e['event']=='tool_call' and not e['ok']:
        print('  !! tool error', e['agent'], e['name'], e['arguments'][:150], e['result'][:150])
