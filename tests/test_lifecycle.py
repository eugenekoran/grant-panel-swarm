"""Offline lifecycle test with a scripted fake LLM (no API calls). Run: python -m tests.test_lifecycle"""
import asyncio
import json
import tempfile
from pathlib import Path

from swarm import run as run_mod
from swarm.config import Config
from swarm.llm import Completion

STATE = {}  # set by the fake to reach the master


class FakeLLM:
    """R1 replies with plain text once; R5 never votes (tests nudges -> abstention); others behave."""
    def __init__(self, cfg, log=None):
        self.cfg, self.n = cfg, 0
        self.plain_done = set()

    async def chat(self, model, messages, tools=None, max_tokens=None, thinking=None):
        await asyncio.sleep(0.01)
        self.n += 1
        if not tools:  # summarizer
            return Completion("summary of earlier work", "", [], "stop", 100, 20, 0.0)
        sysmsg = messages[0]["content"]
        aid = sysmsg.split("You are panelist ")[1].split(" ")[0]
        m = STATE["master"]
        def call(name, **args):
            return Completion("", "thinking...", [{"id": f"c{self.n}", "name": name, "arguments": json.dumps(args)}],
                              "tool_calls", 1000 + 50 * len(messages), 50, 0.0)
        if aid == "R1" and aid not in self.plain_done:
            self.plain_done.add(aid)
            return Completion("I think P1 is good.", "", [], "stop", 1000, 10, 0.0)
        last = messages[-1]
        if last["role"] == "tool" and last["content"].startswith("Turn"):
            pass
        # one action then done
        if messages[-1]["role"] == "tool":
            return call("done")
        for kind, pid in m.obligations(aid):
            if kind == "critique":
                return call("submit_critique", pid=pid, text="ok", importance=3, rigor=3, expertise=2,
                            overall=3 if pid == "P1" else 2)
            if kind == "vote" and aid != "R5":
                return call("vote", pid=pid, overall=4)
        if aid == "R2" and not getattr(self, "msg_sent", False):
            self.msg_sent = True
            return call("post_to_all", text="hello panel")
        if aid == "R3" and not getattr(self, "bad_sent", False):
            self.bad_sent = True
            return call("vote", pid="P9", overall=11)
        if aid == "R4" and not getattr(self, "reported", False):
            self.reported = True
            return call("report_panelist", target="R2", description="harsh on X", evidence="P1 critique")
        if aid == "R4" and not getattr(self, "bad_report", False):
            self.bad_report = True
            return call("report_panelist", target="self", description="d", evidence="e")  # rejected
        if aid == "R2" and not getattr(self, "disclosed", False):
            self.disclosed = True
            return call("disclose_tendency", description="I favor Y", evidence="my P2 scores")
        return call("done")


async def scenario_turn_limit():
    """Every agent hits max_turns=1 while proposals are still in REVIEW: proposals must still lock."""
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(n_agents=5, payline_k=1, max_turns=1, out_dir=d, run_name="t")
        res = await asyncio.wait_for(run_mod.run(cfg), 60)
    print("turn-limit end:", res["end_reason"], [(r["pid"], r["state"], r["abstentions"]) for r in res["ranking"]])
    assert all(r["state"] == "LOCKED" for r in res["ranking"]), res["ranking"]


def unit_checks():
    from swarm.agent import _clean_args
    from swarm.llm import LLM
    assert _clean_args("") == "{}" and _clean_args("{bad") == "{}" and _clean_args("[1]") == "{}"
    assert json.loads(_clean_args('{"pid": "P1"}')) == {"pid": "P1"}
    class Msg:
        content = "secret plan</think>visible"; tool_calls = None; model_extra = {}
    class Ch:
        message = Msg(); finish_reason = "stop"
    class R:
        choices = [Ch()]; usage = None; model_extra = {}
    c = LLM._parse(R(), 0.0)
    assert c.content == "visible" and "secret plan" in c.reasoning, c
    print("unit checks OK")


def unit_discussion():
    """Comments: routing, caps, the FINAL_CALL hold, and grep_proposal visibility."""
    from swarm.log import EventLog
    from swarm.master import FINAL_CALL, VOTING, Master
    from swarm.prompts import status_card
    from swarm.tools import execute
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(n_agents=4, reviewers_per_proposal=2, max_comments_per_proposal=2)
        props = [{"pid": "P1", "title": "T", "abstract": "Short abstract.", "animal": False,
                  "full_text": "Short abstract.\nPower: n = 24 per group.\nSECRET full-text line."}]
        ids = ["R1", "R2", "R3", "R4"]
        m = Master(cfg, props, ids, EventLog(Path(d) / "e.jsonl"))
        p = m.proposals["P1"]
        r1, r2 = p.assigned
        out, rest = (lambda o: (o[0], o[1:]))([a for a in ids if a not in p.assigned])
        run = lambda aid, name, **kw: execute(m, aid, name, json.dumps(kw), 10_000)
        res, ok = run(out, "comment", pid="P1", text="too early")
        assert not ok and "REVIEW" in res, res
        g, _ = run(out, "grep_proposal", pid="P1", pattern="SECRET")
        assert "0 match" in g and "abstract only" in g, g
        g, _ = run(r1, "grep_proposal", pid="P1", pattern="n = \\d+")
        assert "line 2: Power" in g, g
        for r in (r1, r2):
            m.submit_critique(r, "P1", "fine", {"importance": 3, "rigor": 3, "expertise": 3, "overall": 3})
        assert p.state == VOTING
        for a in ids:
            m.take_mail(a)
        res, ok = run(out, "comment", pid="P1", text=f"{r1}, the power analysis looks thin.")
        assert ok and "c1" in res and "1 of 2 comments left" in res, res
        assert all(m.mailbox[a] for a in p.assigned) and not any(m.mailbox[a] for a in rest + [out])
        m.cast_vote(out, "P1", 4)
        for a in rest:
            m.cast_vote(a, "P1", 4)
        assert p.state == VOTING and set(p.reply_pending) == set(p.assigned)  # held: reviewers have not answered
        crit, _ = run(out, "read_critiques", pid="P1")
        assert crit.startswith("PANEL DISCUSSION") and "[c1]" in crit, crit
        for a in p.assigned:
            m.end_turn(a, m.start_turn(a))
        assert p.state == FINAL_CALL, p.state
        res, ok = run(r1, "comment", pid="P1", text="It is fine.", reply_to="c1")
        assert ok and m.mailbox[out] and any(x.comment_id for x in m.mailbox[out])
        m.take_mail(out)
        res, _ = run(r1, "comment", pid="P1", text=f"again, {out}")
        assert "last comment on P1" in res, res
        assert "your comments: all used" in status_card(r1, m)
        assert "used all 2 of your comments on P1" in run(r1, "read_critiques", pid="P1")[0]
        assert "cannot reply" not in m.mailbox[out][-1].text  # out still has 1 comment left
        res, ok = run(out, "comment", pid="P1", text=f"{r1}, last word.")
        assert ok and "cannot reply" in m.mailbox[r1][-1].text, m.mailbox[r1][-1].text
        res, ok = run(r1, "comment", pid="P1", text="third")
        assert not ok and "all 2 of your 2 comments" in res, res
        res, ok = run(r2, "comment", pid="P1", text="x", reply_to="c99")
        assert not ok and "c99" in res, res
    print("discussion checks OK")


async def main():
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(n_agents=5, payline_k=1, max_turns=30, max_nudges=2, out_dir=d, run_name="t",
                     compact_frac=0.05)  # force compaction too
        orig_master = run_mod.Master
        class M(orig_master):
            def __init__(self, *a, **k):
                super().__init__(*a, **k); STATE["master"] = self
        run_mod.Master, run_mod.LLM = M, FakeLLM
        res = await asyncio.wait_for(run_mod.run(cfg), 60)
        events = [json.loads(l) for l in (Path(d) / "t" / "events.jsonl").read_text().splitlines()]
    kinds = {}
    for e in events:
        kinds[e["event"]] = kinds.get(e["event"], 0) + 1
    print(json.dumps(kinds, indent=1))
    print("end:", res["end_reason"])
    for r in res["ranking"]:
        print(r["pid"], r["state"], r["final_score"], r["votes"], r["abstentions"], r["funded"])
    assert res["end_reason"] == "all proposals locked", res["end_reason"]
    assert all(r["state"] == "LOCKED" for r in res["ranking"])
    assert "R5" in res["ranking"][0]["abstentions"] or "R5" in res["ranking"][1]["abstentions"]
    assert kinds.get("invalid_reply", 0) >= 1 and kinds.get("nudge", 0) >= 1 and kinds.get("compaction_end", 0) >= 1
    reps = sorted((r["reporter"], r["kind"], r["target"]) for r in res["reports"])
    assert reps == [("R2", "self", "self"), ("R4", "other", "R2")], reps
    sro = [e["text"] for e in events if e["event"] == "mail_sent" and e["sender"] == "SRO"]
    assert sum("PANEL INTEGRITY" in t for t in sro) == cfg.n_agents  # in every kickoff
    reminded = {t.split(".")[0] for t in sro if "Integrity reminder" in t}
    assert len(reminded) == 1 and next(iter(reminded)).startswith("Voting open on"), reminded  # first opening only
    await scenario_turn_limit()
    unit_checks()
    unit_discussion()
    print("OK")

asyncio.run(main())
