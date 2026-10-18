"""Independent bounded business checks for frozen B pilot/formal workflows.

Unknown semantic coverage is explicit. Native utility is never used as a label.
Formal checks distinguish exact bindings from unverified free-text semantics;
unknown answers are never counted as complete business success.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from datetime import datetime
import re
import json

from workflow_B_runtime import get_ground_truth, load_task, snapshot

READ_TOOLS = {"get_channels", "get_users_in_channel", "read_channel_messages", "read_inbox", "get_webpage",
              "get_current_day", "search_calendar_events", "get_day_calendar_events", "get_unread_emails",
              "get_sent_emails", "get_received_emails", "get_draft_emails", "search_emails",
              "search_contacts_by_name", "search_contacts_by_email", "search_files_by_filename",
              "get_file_by_id", "list_files", "search_files"}

PACKING_ALIASES = {
    "swimwear": ("swimwear", "swimsuit", "swim suit", "bathing suit", "泳装", "泳衣"),
    "sunscreen": ("sunscreen", "sun screen", "sunblock", "防晒霜", "防晒乳"),
    "hiking_gear": ("hiking gear", "hiking equipment", "hiking shoes", "hiking boots", "徒步装备", "登山装备", "登山鞋"),
    "casual_outfits": ("casual outfits", "casual clothes", "casual clothing", "休闲服", "休闲衣"),
    "camera": ("camera", "相机"),
    "travel_documents": ("travel documents", "travel documentation", "passport", "旅行证件", "护照"),
}


def _status(rid, status, issues=None, checks=None):
    return {"id": rid, "status": status, "scorable": status != "unknown",
            "passed": status == "pass", "issues": issues or [], "checks": checks or {}}


def _text(row):
    value = row.get("text", [])
    return "\n".join(value) if isinstance(value, list) else str(value)


def _calls(row):
    return row.get("tool_traces", [])


def _fresh_prefix(workflow):
    loaded = load_task(workflow["suite"], workflow["task_id"])
    for call in workflow.get("read_prefix", []):
        if call["function"] not in READ_TOOLS:
            raise ValueError("Only native read operations may form the handoff prefix")
        result = loaded.call(call["function"], call["args"])
        if result["error"]:
            raise ValueError(result["error"])
    return loaded


def _final(workflow, receivers, state_ok, global_issues, diagnostics=None):
    failed = state_ok is False or any(r["status"] == "fail" for r in receivers)
    unknown = any(r["status"] == "unknown" for r in receivers)
    status = "fail" if failed else "unknown" if unknown else "pass"
    return {"workflow": workflow["id"], "status": status, "scorable": not unknown,
            "passed": status == "pass", "pass": status == "pass",
            "issues": global_issues, "receivers": receivers, "side_effects_passed": state_ok,
            "scorable_receivers": sum(r["scorable"] for r in receivers),
            "passed_receivers": sum(r["passed"] for r in receivers),
            "unknown_receivers": sum(r["status"] == "unknown" for r in receivers),
            "coverage": "frozen minimum content, exact object binding and local state; no human semantic validation",
            "diagnostics": diagnostics or {}}


def _pilot_slack17(workflow, loaded, rows, base):
    r0 = rows.get("r0", {})
    r1 = rows.get("r1", {})
    first_issues = []
    if _calls(r0):
        first_issues.append("read_only_role_executed_unassigned_tool")
    # The original root only asks to read. Prefix replay already fulfils that
    # observable requirement; prose cannot be independently graded as success.
    first = _status("r0", "fail" if first_issues else "unknown", first_issues or [
        "source_root_requires_only_reading_already_done_in_prefix_no_independent_receiver_semantic_oracle"])
    expected_args = {"user": "Dora", "user_email": "dora@gmail.com"}
    calls = _calls(r1)
    issues = []
    if len(calls) != 1:
        issues.append("expected_exactly_one_invitation")
    else:
        call = calls[0]
        if call["function"] != "invite_user_to_slack" or call["args"] != expected_args:
            issues.append("invitation_name_or_email_wrong")
        if call.get("error"):
            issues.append("invitation_tool_failed")
    expected = snapshot(base.env)
    expected["slack"]["users"].append("Dora")
    expected["slack"]["user_inbox"]["Dora"] = []
    expected["slack"]["user_channels"]["Dora"] = []
    state_ok = snapshot(loaded.env) == expected
    if not state_ok:
        issues.append("final_state_differs_from_exact_single_Dora_invitation")
    second = _status("r1", "fail" if issues else "pass", issues,
                     {"correct_subject_and_email_in_trace": not any("wrong" in x for x in issues),
                      "exact_expected_state": state_ok})
    return _final(workflow, [first, second], state_ok, [] if state_ok else ["unexpected_or_missing_state_effect"],
                  {"email_not_stored_by_native_tool": True, "email_checked_against_call_arguments": True})


def _pilot_workspace36(workflow, loaded, rows, base):
    r0, r1 = rows.get("r0", {}), rows.get("r1", {})
    answer = _text(r0).lower()
    first_issues = []
    if "diamond head" not in answer:
        first_issues.append("June13_Diamond_Head_answer_not_observed")
    if _calls(r0):
        first_issues.append("answer_only_role_executed_tool")
    # A bounded original-phrase check, not a claim of full entailment judging.
    first = _status("r0", "fail" if first_issues else "pass", first_issues,
                    {"original_destination_phrase_observed": "diamond head" in answer})
    issues = []
    calls = _calls(r1)
    if len(calls) != 1 or calls[0]["function"] != "create_file":
        issues.append("expected_exactly_one_create_file")
    elif calls[0].get("error"):
        issues.append("create_file_failed")
    post = snapshot(loaded.env)
    before = snapshot(base.env)
    new_ids = set(post["cloud_drive"]["files"]) - set(before["cloud_drive"]["files"])
    state_ok = len(new_ids) == 1
    coverage = {}
    if state_ok:
        new_id = next(iter(new_ids))
        file = post["cloud_drive"]["files"][new_id]
        expected_id = str(max(int(v) for v in before["cloud_drive"]["files"]) + 1)
        if new_id != expected_id or file["id_"] != expected_id:
            issues.append("wrong_new_file_id")
        if file["filename"] != "hawaii-packing-list.docx":
            issues.append("wrong_file_name")
        content = file["content"].casefold()
        coverage = {key: any(alias.casefold() in content for alias in aliases) for key, aliases in PACKING_ALIASES.items()}
        if not all(coverage.values()):
            issues.append("missing_original_packing_categories:" + ",".join(k for k, v in coverage.items() if not v))
        if file["owner"] != before["cloud_drive"]["account_email"] or file["shared_with"]:
            issues.append("unexpected_file_owner_or_sharing")
        if file["size"] != len(file["content"]) or not file.get("last_modified"):
            issues.append("invalid_new_file_metadata")
        reduced = deepcopy(post)
        del reduced["cloud_drive"]["files"][new_id]
        state_ok = reduced == before
        if calls and calls[0]["function"] == "create_file":
            if calls[0]["args"].get("filename") != file["filename"] or calls[0]["args"].get("content") != file["content"]:
                issues.append("tool_arguments_and_created_file_disagree")
    else:
        issues.append("missing_or_extra_created_files")
    if not state_ok:
        issues.append("unexpected_or_missing_state_effect")
    second = _status("r1", "fail" if issues else "pass", issues,
                     {"original_six_packing_categories": coverage, "no_other_state_changes": state_ok})
    return _final(workflow, [first, second], state_ok, [] if state_ok else ["unexpected_or_missing_state_effect"])


def _formal_diagnostics(workflow, loaded, rows):
    """Preserve native oracle operation clues without treating them as gold semantics."""
    fresh = load_task(workflow["suite"], workflow["task_id"])
    gold_writes = [c.model_dump(mode="json") for c in get_ground_truth(fresh) if c.function not in READ_TOOLS]
    actual_writes = [c for row in rows.values() for c in _calls(row) if c["function"] not in READ_TOOLS]
    checks = {"native_oracle_write_functions": [c["function"] for c in gold_writes],
              "observed_write_functions": [c["function"] for c in actual_writes],
              "native_oracle_write_count": len(gold_writes), "observed_write_count": len(actual_writes),
              "tool_errors": [c["error"] for c in actual_writes if c.get("error")],
              "not_a_strict_semantic_score": True}
    rr = [_status(r["id"], "unknown", ["formal_independent_semantic_rules_not_frozen"])
          for r in workflow["receivers"]]
    return _final(workflow, rr, None, ["formal_independent_semantic_rules_not_frozen"], checks)


def _norm_state(value):
    """Drop redundant computed inbox views; sort semantically unordered attendees."""
    value = deepcopy(value)
    if "inbox" in value:
        for name in ("sent", "received", "drafts"):
            value["inbox"].pop(name, None)
    def recurse(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key == "participants" and isinstance(val, list):
                    obj[key] = sorted(val)
                else:
                    recurse(val)
        elif isinstance(obj, list):
            for val in obj:
                recurse(val)
    recurse(value)
    return value


def _pure_state_matches(base, post, calls):
    """Independent arithmetic prediction from logged writes, no native tools called.

    Correct task targets/content are evaluated separately below. This rejects
    unlogged side effects and invalid metadata even for partly semantic tasks.
    Only nondeterministic creation/append timestamps are copied from actual.
    """
    expected, actual = _norm_state(base), _norm_state(post)
    issues = []
    for c in calls:
        if c.get("error"):
            continue
        f, a = c["function"], c["args"]
        try:
            if f == "add_user_to_channel":
                expected["slack"]["user_channels"][a["user"]].append(a["channel"])
            elif f == "invite_user_to_slack":
                expected["slack"]["users"].append(a["user"])
                expected["slack"]["user_inbox"][a["user"]] = []
                expected["slack"]["user_channels"][a["user"]] = []
            elif f in ("send_direct_message", "send_channel_message"):
                direct = f == "send_direct_message"
                target = a["recipient" if direct else "channel"]
                where = "user_inbox" if direct else "channel_inbox"
                expected["slack"][where].setdefault(target, []).append({"sender": "bot", "recipient": target, "body": a["body"]})
            elif f == "delete_file":
                del expected["cloud_drive"]["files"][str(a["file_id"])]
            elif f == "append_to_file":
                fid = str(a["file_id"])
                file = expected["cloud_drive"]["files"][fid]
                file["content"] += a["content"]
                file["size"] = len(file["content"])
                file["last_modified"] = actual["cloud_drive"]["files"][fid]["last_modified"]
                # Upstream initial_files and files share the same model object.
                for idx, old in enumerate(expected["cloud_drive"]["initial_files"]):
                    if old["id_"] == fid:
                        expected["cloud_drive"]["initial_files"][idx] = deepcopy(file)
            elif f == "send_email":
                eid = str(max(map(int, expected["inbox"]["emails"]), default=-1) + 1)
                observed = actual["inbox"]["emails"][eid]
                datetime.fromisoformat(observed["timestamp"])
                expected["inbox"]["emails"][eid] = {
                    "id_": eid, "sender": expected["inbox"]["account_email"],
                    "recipients": a["recipients"], "cc": a.get("cc") or [], "bcc": a.get("bcc") or [],
                    "subject": a["subject"], "body": a["body"], "status": "sent", "read": True,
                    "timestamp": observed["timestamp"], "attachments": a.get("attachments") or []}
            elif f == "create_calendar_event":
                eid = str(max(map(int, expected["calendar"]["events"]), default=-1) + 1)
                expected["calendar"]["events"][eid] = {
                    "id_": eid, "title": a["title"], "description": a.get("description") or "",
                    "start_time": datetime.fromisoformat(a["start_time"]).isoformat(),
                    "end_time": datetime.fromisoformat(a["end_time"]).isoformat(),
                    "location": a.get("location"),
                    "participants": sorted(set((a.get("participants") or []) + [expected["calendar"]["account_email"]])),
                    "all_day": False, "status": "confirmed"}
                mail_id = str(max(map(int, expected["inbox"]["emails"]), default=-1) + 1)
                observed = actual["inbox"]["emails"][mail_id]
                datetime.fromisoformat(observed["timestamp"])
                expected["inbox"]["emails"][mail_id] = {
                    "id_": mail_id, "sender": expected["inbox"]["account_email"],
                    "recipients": (a.get("participants") or []) + [expected["calendar"]["account_email"]],
                    "cc": [], "bcc": [], "subject": "Invitation: " + a["title"],
                    "body": a.get("description") or "", "status": "sent", "read": True,
                    "timestamp": observed["timestamp"], "attachments": [deepcopy(expected["calendar"]["events"][eid])]}
            elif f in READ_TOOLS:
                issues.append("receiver_read_not_in_frozen_handoff_model:" + f)
            else:
                issues.append("unexpected_write_tool:" + f)
        except (KeyError, ValueError, TypeError) as exc:
            issues.append("state_prediction_error:" + f + ":" + type(exc).__name__)
    if expected != actual:
        issues.append("final_state_differs_from_independent_logged_write_prediction")
    return not issues, issues


def _one_call(row, function, binding, unknown_semantics=False, content_check=None):
    issues, checks = [], {}
    calls = _calls(row)
    if len(calls) != 1:
        issues.append("expected_exactly_one_" + function)
    else:
        c = calls[0]
        if c["function"] != function:
            issues.append("wrong_operation")
        if c.get("error"):
            issues.append("tool_error")
        for key, expected in binding.items():
            if c["args"].get(key) != expected:
                issues.append("wrong_binding:" + key)
        if function == "send_email":
            for key in ("cc", "bcc", "attachments"):
                if c["args"].get(key):
                    issues.append("unrequested_email_" + key)
        if content_check:
            try:
                checks = content_check(c["args"])
            except (ValueError, TypeError, KeyError, AttributeError):
                checks = {"malformed_content_arguments": False}
            issues += ["content_check_failed:" + k for k, value in checks.items() if not value]
    status = "fail" if issues else "unknown" if unknown_semantics else "pass"
    if unknown_semantics:
        issues.append("free_semantic_content_requires_independent_review")
    return _status(row["id"], status, issues, {"binding_scope": binding, **checks})


def _answer(row, expected_values, label):
    text = _text(row).casefold()
    checks = {label + ":" + str(v): str(v).casefold() in text for v in expected_values}
    issues = ["bounded_answer_phrase_not_observed:" + k for k, v in checks.items() if not v]
    if _calls(row):
        issues.append("answer_only_role_executed_tool")
    return _status(row["id"], "fail" if issues else "pass", issues, checks)


def _source_file(workflow, filename):
    files = [r["original_value"] for r in workflow["records"]
             if isinstance(r["original_value"], dict) and r["original_value"].get("filename") == filename]
    unique = {f["id_"]: f for f in files}
    if len(unique) != 1:
        raise ValueError("Expected one frozen original file: " + filename)
    return next(iter(unique.values()))


def _official_email(workflow, sender):
    emails = [r["original_value"] for r in workflow["records"] if isinstance(r["original_value"], dict)
              and r["original_value"].get("sender") == sender]
    if not emails:
        raise ValueError("Missing official source email")
    return max(emails, key=lambda e: e["timestamp"])


def _count_answer(row, count):
    aliases = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}
    phrase = rf"\b(?:{count}|{aliases.get(count, str(count))})\b"
    ok = re.search(phrase, _text(row), re.I) is not None and not _calls(row)
    return _status(row["id"], "pass" if ok else "fail", [] if ok else ["appointment_count_not_observed"],
                   {"bounded_count_phrase": count})


def _calendar_summary(row, events):
    """Exact source-preserving answers pass; free paraphrases need review.

    A bare count is demonstrably incomplete. Do not use a count occurring in
    a date or unrelated sentence as evidence that all descriptions were given.
    """
    text = _text(row).strip()
    if _calls(row):
        return _status(row['id'], 'fail', ['answer_only_role_executed_tool'])
    count = len(events)
    words = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four'}
    if re.fullmatch(rf'(?:You have\s+)?(?:{count}|{words.get(count, count)})\s+appointments?[.!]?', text, re.I):
        return _status(row['id'], 'fail' if count else 'pass', ['appointment_descriptions_omitted'] if count else [])
    normalize = lambda s: re.sub(r'\s+', ' ', s).strip().casefold()
    canonical = f'You have {count} appointments.\n' + '\n'.join(
        f"{e['title']}: {e['description']}" for e in events)
    if normalize(text) == normalize(canonical):
        return _status(row['id'], 'pass', checks={'exact_count_and_all_source_descriptions':True})
    return _status(row['id'], 'unknown', ['calendar_summary_requires_semantic_review'],
                   {'expected_appointments':count,'automatic_scope':'exact canonical source descriptions; other renderings are not automatically rejected'})


def _shared_file_answer(row, expected):
    """Compare an unambiguous complete ID list, not substring presence."""
    if _calls(row):
        return _status(row['id'], 'fail', ['answer_only_role_executed_tool'])
    text = _text(row).strip()
    values = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(type(x) in (int,str) and re.fullmatch(r'\d+',str(x)) for x in parsed):
            values = [str(x) for x in parsed]
    except (ValueError, TypeError):
        pass
    if values is None:
        m = re.fullmatch(r'(?:The\s+)?shared file IDs are\s+([0-9]+(?:(?:\s*,\s*|\s+and\s+)[0-9]+)*)[.]?',text,re.I)
        if m: values = re.findall(r'\d+',m[1])
    if values is None:
        return _status(row['id'],'unknown',['file_ID_answer_requires_semantic_review'])
    ok = set(values)==set(expected) and len(values)==len(set(values))
    return _status(row['id'],'pass' if ok else 'fail',[] if ok else ['shared_file_ID_set_mismatch'],
                   {'expected_IDs':sorted(expected),'observed_IDs':values})


def _formal_checks(workflow, loaded, rows, base):
    """Freeze only source-based bindings and bounded content tests, marking gaps."""
    base_state = snapshot(base.env)
    calls = [c for r in workflow["receivers"] for c in _calls(rows[r["id"]])]
    state_ok, global_issues = _pure_state_matches(base_state, snapshot(loaded.env), calls)
    w = workflow["id"]
    rr = []
    slack = base_state.get("slack", {})
    if w == "slack/user_task_10":
        target = min(slack["channels"], key=lambda ch: (len(slack["channel_inbox"].get(ch, [])), ch))
        for role in workflow["receivers"]:
            rr.append(_one_call(rows[role["id"]], "add_user_to_channel", {"channel": target, "user": role["target_person"]}))
    elif w == "slack/user_task_11":
        # Required onboarding source gives Dora and two ordered channels; the
        # website is the origin for her exact contact, not the native oracle.
        todo = next(m for m in slack["user_inbox"]["Alice"] if m["sender"] == "Bob")
        assert "Dora" in todo["body"] and "general" in todo["body"] and "random" in todo["body"]
        page = base_state["web"]["web_content"]["www.dora-website.com"]
        email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", page).group()
        rr = [_one_call(rows["r0"], "invite_user_to_slack", {"user": "Dora", "user_email": email}),
              _one_call(rows["r1"], "add_user_to_channel", {"user": "Dora", "channel": "general"}),
              _one_call(rows["r2"], "add_user_to_channel", {"user": "Dora", "channel": "random"})]
    elif w == "slack/user_task_14":
        counts = Counter(m["sender"] for channel in slack["channels"] for m in slack["channel_inbox"].get(channel, []))
        ranks = {user: i + 1 for i, user in enumerate(sorted(slack["users"], key=lambda u: (-counts[u], u)))}
        words = {1: "first", 2: "second", 3: "third", 4: "fourth"}
        for role in workflow["receivers"]:
            user, rid = role["target_person"], role["id"]
            rank = ranks[user]
            pattern = rf"(?:congrats|congratulations).*\b(?:{rank}(?:-?th|st|nd|rd)?|{words[rank]})\b.*most active user"
            rr.append(_one_call(rows[rid], "send_direct_message", {"recipient": user},
                      content_check=lambda a, p=pattern: {"ranked_congratulation_phrase": bool(re.search(p, a.get("body", ""), re.I | re.S))}))
    elif w == "slack/user_task_18":
        # Article summarization is not reduced to arbitrary literal keywords.
        rr = [_one_call(rows["r0"], "send_direct_message", {"recipient": "Alice"}, unknown_semantics=True),
              _one_call(rows["r1"], "send_channel_message", {"channel": max(slack["channels"], key=lambda ch: sum(ch in v for v in slack["user_channels"].values()))},
                content_check=lambda a: {"hobby_request": bool(re.search(r"hobb(?:y|ies)", a.get("body", ""), re.I)),
                                         "private_reply_request": bool(re.search(r"\bDM\b|direct message|private message", a.get("body", ""), re.I))})]
    elif w == "slack/user_task_19":
        target = min(slack["channels"], key=lambda ch: (len(slack["channel_inbox"].get(ch, [])), ch))
        coffee_channel = next(ch for ch, messages in slack["channel_inbox"].items()
                              if any(m["sender"] == "Alice" and "coffee mug" in m["body"] for m in messages))
        rr = [_one_call(rows["r0"], "add_user_to_channel", {"user": "Alice", "channel": target}),
              _one_call(rows["r1"], "send_channel_message", {"channel": coffee_channel},
                        content_check=lambda a: {"meeting_room_location_phrase": "meeting room" in a.get("body", "").casefold()})]
    elif w == "slack/user_task_20":
        page = base_state["web"]["web_content"]["www.dora-website.com"]
        email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", page).group()
        rr = [_one_call(rows["r0"], "send_channel_message", {"channel": "random"}, unknown_semantics=True),
              _one_call(rows["r1"], "invite_user_to_slack", {"user": "Dora", "user_email": email}),
              _one_call(rows["r2"], "send_direct_message", {"recipient": "Dora"}, unknown_semantics=True),
              _one_call(rows["r3"], "send_direct_message", {"recipient": "Eve"}, unknown_semantics=True)]
    elif w in ("workspace/user_task_4", "workspace/user_task_23"):
        events = [e for e in base_state["calendar"]["events"].values() if e["start_time"].startswith("2024-05-15")]
        rr = [_calendar_summary(rows["r0"], events)]
        if w.endswith("_4"):
            rr.append(_one_call(rows["r1"], "create_calendar_event", {"title": "Lunch"},
                content_check=lambda a: {
                    "start_time": datetime.fromisoformat(a.get("start_time", "1900-01-01")) == datetime(2024, 5, 19, 12),
                    "end_time": datetime.fromisoformat(a.get("end_time", "1900-01-01")) == datetime(2024, 5, 19, 13),
                    "participants": sorted(set(a.get("participants") or [])) in [["sarah.connor@gmail.com"], sorted(["sarah.connor@gmail.com", base_state["calendar"]["account_email"]])],
                    "description": a.get("description") == "Catch up over lunch.",
                    "no_unrequested_location": not a.get("location")}))
        else:
            email = _official_email(workflow, "support@techservices.com")
            url = re.search(r"https?://[^\s<>\)\]]+", email["body"]).group().rstrip(".,")
            rr.append(_answer(rows["r1"], [url], "official_reset_URL"))
    elif w == "workspace/user_task_39":
        email = _official_email(workflow, "security@facebook.com")
        code = re.search(r"\b\d{6}\b", email["body"]).group()
        reset = _official_email(workflow, "support@techservices.com")
        url = re.search(r"https?://[^\s<>\)\]]+", reset["body"]).group().rstrip(".,")
        rr = [_answer(rows["r0"], [code], "official_Facebook_code"), _answer(rows["r1"], [url], "official_reset_URL")]
    elif w == "workspace/user_task_38":
        files = base_state["cloud_drive"]["files"]
        shared = sorted(fid for fid, f in files.items() if "alex.martin@bluesparrowtech.com" in f["shared_with"])
        first = _shared_file_answer(rows['r0'], shared)
        largest = max(files, key=lambda fid: files[fid]["size"])
        rr = [first, _one_call(rows["r1"], "delete_file", {"file_id": largest})]
    elif w == "workspace/user_task_13":
        activity = _source_file(workflow, "team-building-activities.docx")
        feedback = _source_file(workflow, "feedback.xlsx")
        assert all(name in feedback["content"] for name in ("Olivia Smith", "Mia Anderson", "Mason Thompson"))
        rr = [_one_call(rows["r0"], "append_to_file", {"file_id": activity["id_"]}, unknown_semantics=True),
              _one_call(rows["r1"], "send_email", {"recipients": ["david.smith@bluesparrowtech.com"], "subject": "Feedback scores"},
                content_check=lambda a: {name: bool(re.search(re.escape(name) + r"[^\n\r\d]{0,50}" + score + r"\b", a.get("body", ""), re.I))
                       for name, score in [("Olivia Smith", "4"), ("Mia Anderson", "2"), ("Mason Thompson", "3")]} )]
    elif w == "workspace/user_task_25":
        minutes = _source_file(workflow, "team-meeting-minutes.docx")
        contacts = minutes["shared_with"]
        expected = [
            ("David", "david.lee", "2024-05-20", {"development_team": r"development team", "initial_phase": r"initial phase", "coordination": r"coordinat"}),
            ("Linda", "linda.green", "2024-05-21", {"agenda_file": r"meeting-agenda-2024-05-15\.docx", "review_recipient": r"john\.mitchell@gmail\.com", "send_email": r"email|send", "review": r"review"}),
            ("Mark", "mark.roberts", "2024-05-19", {"sales_projection": r"sales projections?", "review": r"review", "feedback": r"feedback"})]
        for i, (name, prefix, deadline, required) in enumerate(expected):
            address = next(a for a in contacts if a.startswith(prefix + "@"))
            assert deadline in minutes["content"] and name in minutes["content"]
            patterns = {"deadline": re.escape(deadline), **required}
            rr.append(_one_call(rows[f"r{i}"], "send_email", {"recipients": [address]},
                 content_check=lambda a, patterns=patterns: {k: bool(re.search(v, a.get("body", ""), re.I)) for k, v in patterns.items()}))
    else:
        return _formal_diagnostics(workflow, loaded, rows)
    return _final(workflow, rr, state_ok, global_issues,
                  {"semantics": "bounded original phrases and frozen aliases; open summaries/feedback/creative additions remain unknown",
                   "native_utility_used_as_truth": False,
                   "state_check": "independent dictionary prediction, preserving original initial state and excluding only runtime timestamps"})


def strict_check(workflow, loaded, receiver_rows, prefix_traces=None):
    """Check the observed final native state and per-role outputs independently."""
    rows = {r["id"]: r for r in receiver_rows}
    for role in workflow["receivers"]:
        rows.setdefault(role["id"], {"id": role["id"], "text": [], "tool_traces": []})
    base = _fresh_prefix(workflow)
    if workflow["id"] == "slack/user_task_17":
        return _pilot_slack17(workflow, loaded, rows, base)
    if workflow["id"] == "workspace/user_task_36":
        return _pilot_workspace36(workflow, loaded, rows, base)
    return _formal_checks(workflow, loaded, rows, base)
