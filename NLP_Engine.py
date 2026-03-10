import json
import spacy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from typing import Optional


try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")


TASK_TRIGGER_VERBS = {
    "fix", "update", "write", "design", "build", "create", "implement",
    "develop", "test", "review", "optimize", "refactor", "deploy", "check",
    "prepare", "complete", "finish", "add", "remove", "document", "resolve",
    "investigate", "improve", "handle", "write", "start", "take", "look",
    "train", "migrate", "scale", "setup", "set", "configure", "integrate",
    "monitor", "automate", "analyse", "analyze", "evaluate", "research",
}

MODAL_TASK_SIGNALS = [
    ["must", "fix"], ["must", "be", "fixed"], ["must", "be", "done"],
    ["will", "handle"], ["will", "write"], ["will", "fix"], ["will", "take"],
    ["will", "prepare"], ["will", "design"], ["will", "test"],
    ["can", "start"], ["try", "optimizing"], ["try", "to"],
    ["take", "care"], ["please", "fix"], ["please", "write"],
    ["please", "prepare"], ["please", "take"], ["please", "handle"],
    ["please", "design"], ["please", "look"], ["look", "into"],
    ["please", "resolve"], ["please", "optimize"],
]

TASK_SIGNAL_PHRASES = [
    ["need", "to"], ["needs", "to"], ["we", "need"], ["someone", "should"],
    ["have", "to"], ["must"], ["should", "also"], ["we", "need", "integration"],
    ["need", "tests"], ["need", "test"], ["someone", "needs"],
]

DEADLINE_ONLY_PHRASES = [
    ["needs", "to", "done"], ["needs", "to", "be", "done"],
    ["done", "by"], ["wait", "until"], ["can", "wait"],
    ["plan", "this", "for"], ["tackle", "this", "by"], ["let", "plan"],
]

ATTRIBUTION_PHRASES = [
    ["worked", "on"], ["you", "worked"], ["since", "you"],
    ["sounds", "like"], ["this", "sounds"], ["this", "probably"],
    ["has", "been", "fixed"], ["has", "been", "resolved"],
    ["already", "fixed"], ["already", "done"], ["issue", "resolved"],
]

FILLER_LEMMAS = {"need", "want", "let", "come", "go", "say", "tell", "try"}

PRIORITY_KEYWORDS = {
    "critical":     "Critical",
    "blocking":     "Critical",
    "blocked":      "Critical",
    "urgent":       "Critical",
    "immediately":  "Critical",
    "asap":         "Critical",
    "high":         "High",
    "important":    "High",
    "priority":     "High",
    "release":      "High",
    "medium":       "Medium",
    "moderate":     "Medium",
    "low":          "Low",
    "eventually":   "Low",
}

DEADLINE_PATTERNS = [
    (["tomorrow", "evening"], "Tomorrow evening"),
    (["tomorrow"],            "Tomorrow"),
    (["end", "week"],         "End of this week"),
    (["end", "of", "week"],   "End of this week"),
    (["friday"],              "Friday"),
    (["next", "monday"],      "Next Monday"),
    (["wednesday"],           "Wednesday"),
    (["next", "week"],        "Next week"),
    (["next", "sprint"],      "Next sprint"),
    (["today"],               "Today"),
    (["immediately"],         "Immediately"),
    (["eod"],                 "End of day"),
]

DEPENDENCY_SIGNALS = {"depends", "dependent", "completion", "resolved", "fixed"}

ONCE_PATTERN = ["once", "after", "when"]

ROLE_MISMATCH = {
    "frontend": {"test", "testing", "automation", "database", "backend", "server"},
    "backend":  {"design", "figma", "onboarding", "screens", "wireframe"},
    "designer": {"test", "testing", "automation", "database", "backend", "server"},
    "qa":       {"design", "figma", "screens", "wireframe"},
}


@dataclass
class TeamMember:
    name:   str
    role:   str
    skills: list[str] = field(default_factory=list)

    def skill_overlap_score(self, description: str) -> int:
        desc_tokens = set(description.lower().split())
        score = 0
        for skill in self.skills:
            skill_tokens = skill.lower().split()
            if all(word in desc_tokens for word in skill_tokens):
                score += 2
            elif any(word in desc_tokens for word in skill_tokens if len(word) > 3):
                score += 1
        return score

    def __repr__(self):
        return f"TeamMember(name='{self.name}', role='{self.role}')"


@dataclass
class Task:
    id:              int
    description:     str
    assigned_to:     str  = "Unassigned"
    deadline:        str  = "TBD"
    priority:        str  = "Medium"
    has_dependency:  bool = False
    dependency_note: str  = ""
    reason:          str  = ""

    def is_assigned(self) -> bool:
        return self.assigned_to != "Unassigned"

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "description":     self.description,
            "assigned_to":     self.assigned_to,
            "deadline":        self.deadline,
            "priority":        self.priority,
            "has_dependency":  self.has_dependency,
            "dependency_note": self.dependency_note,
            "reason":          self.reason,
        }

    def __repr__(self):
        return f"Task(id={self.id}, to='{self.assigned_to}', priority='{self.priority}')"


class BaseNLPComponent(ABC):

    def _tokens_contain(self, tokens: list[str], phrase: list[str]) -> bool:
        token_set = set(t.lower() for t in tokens)
        return all(word in token_set for word in phrase)

    def _get_tokens(self, sent) -> list[str]:
        return [token.text for token in sent]

    @abstractmethod
    def process(self, *args, **kwargs):
        pass


class TaskExtractor(BaseNLPComponent):

    def __init__(self, team_members: list[TeamMember]):
        self._team       = team_members
        self._team_names = [m.name for m in team_members]


    def _is_deadline_only(self, tokens: list[str]) -> bool:
        lower = [t.lower() for t in tokens]
        return any(self._tokens_contain(lower, p) for p in DEADLINE_ONLY_PHRASES)

    def _is_attribution(self, tokens: list[str]) -> bool:
        lower = [t.lower() for t in tokens]
        return any(self._tokens_contain(lower, p) for p in ATTRIBUTION_PHRASES)

    def _has_action_verb(self, sent) -> bool:
        return any(
            t.pos_ == "VERB" and t.lemma_.lower() in TASK_TRIGGER_VERBS
            for t in sent
        )

    def _has_modal_task_signal(self, tokens: list[str]) -> bool:
        lower = [t.lower() for t in tokens]
        return any(self._tokens_contain(lower, p) for p in MODAL_TASK_SIGNALS)

    def _is_task_sentence(self, sent) -> bool:
        tokens = self._get_tokens(sent)
        lower  = [t.lower() for t in tokens]

        if self._is_deadline_only(tokens):
            return False

        if self._is_attribution(tokens):
            return False

        if len([t for t in sent if not t.is_space]) < 3:
            return False

        if self._has_modal_task_signal(tokens):
            return True

        if not self._has_action_verb(sent):
            return False

        for token in sent:
            if token.dep_ == "ROOT" and token.lemma_.lower() in TASK_TRIGGER_VERBS:
                return True

        return any(self._tokens_contain(lower, p) for p in TASK_SIGNAL_PHRASES)


    def _extract_assignee(self, sent) -> Optional[str]:
        for ent in sent.as_doc().ents:
            if ent.label_ == "PERSON" and ent.text in self._team_names:
                return ent.text
        sent_tokens = {token.text for token in sent}
        return next((n for n in self._team_names if n in sent_tokens), None)

    def _peek_assignee(self, sents: list, idx: int) -> Optional[str]:
        BACK_REFERENCE_TOKENS = {"this", "that", "it", "these", "those"}

        if idx + 1 < len(sents):
            next_sent   = sents[idx + 1]
            next_tokens = self._get_tokens(next_sent)
            next_lower  = {t.lower() for t in next_tokens}

            if self._is_attribution(next_tokens) or not self._is_task_sentence(next_sent):
                if next_lower.intersection(BACK_REFERENCE_TOKENS):
                    return self._extract_assignee(next_sent)

        return None

    def _extract_deadline(self, tokens: list[str], next_tokens: list[str] = None) -> Optional[str]:
        lowered = [t.lower() for t in tokens]
        for pattern, label in DEADLINE_PATTERNS:
            if self._tokens_contain(lowered, pattern):
                return label
        if next_tokens:
            lowered_next = [t.lower() for t in next_tokens]
            for pattern, label in DEADLINE_PATTERNS:
                if self._tokens_contain(lowered_next, pattern):
                    return label
        return None

    DEADLINE_PRIORITY_MAP = {
        "Today":            "Critical",
        "Immediately":      "Critical",
        "End of day":       "Critical",
        "Tomorrow evening": "High",
        "Tomorrow":         "High",
        "Friday":           "High",
        "End of this week": "High",
        "Wednesday":        "Medium",
        "Next Monday":      "Medium",
        "Next week":        "Medium",
        "Next sprint":      "Low",
        "TBD":              "Low",
    }

    def _extract_priority(self, tokens: list[str], deadline: str = None) -> str:
        lowered = set(t.lower() for t in tokens)
        for keyword, priority in PRIORITY_KEYWORDS.items():
            if keyword in lowered:
                return priority
        if deadline:
            return self.DEADLINE_PRIORITY_MAP.get(deadline, "Medium")
        return "Medium"

    SKIP_STARTERS = {"once", "after", "when", "and", "but", "so", "also"}

    def _extract_description(self, sent, assignee: Optional[str]) -> str:
        skip      = {t.i for t in sent if assignee and t.text == assignee}
        keep_pos  = {"VERB", "NOUN", "PROPN", "ADJ", "ADP", "PART", "NUM"}
        skip_deps = {"punct", "cc", "mark"}
        parts = [
            t.text for t in sent
            if t.i not in skip
            and t.dep_ not in skip_deps
            and t.pos_ in keep_pos
            and t.lemma_.lower() not in FILLER_LEMMAS
        ]

        while parts and parts[0].lower() in self.SKIP_STARTERS:
            parts.pop(0)

        desc = " ".join(parts).strip()
        return (desc[0].upper() + desc[1:]) if desc else sent.text.strip()

    def _check_dependency(self, tokens: list[str]) -> tuple[bool, str]:
        lowered = set(t.lower() for t in tokens)

        if lowered.intersection(DEPENDENCY_SIGNALS):
            for token in tokens:
                clean = token.lstrip("#")
                if clean.isdigit():
                    return True, f"Depends on Task #{clean}"
            return True, "Has dependency (see transcript)"

        lower_list = [t.lower() for t in tokens]
        for word in ONCE_PATTERN:
            if word in lowered:
                return True, f"Depends on prior task completion"

        return False, ""

    def _member_reason(self, name: str) -> str:
        for m in self._team:
            if m.name == name:
                return f"{m.role} — {', '.join(m.skills[:2])}" if m.skills else m.role
        return ""


    def process(self, transcript: str) -> list[Task]:
        doc   = _NLP(transcript)
        sents = list(doc.sents)
        tasks = []
        task_id       = 1
        last_assignee = None

        for idx, sent in enumerate(sents):
            tokens = self._get_tokens(sent)

            if not self._is_task_sentence(sent):
                a = self._extract_assignee(sent)
                if a:
                    last_assignee = a
                continue

            assignee = self._extract_assignee(sent)

            if not assignee:
                assignee = self._peek_assignee(sents, idx)

            if not assignee and last_assignee:
                if idx > 0:
                    prev_tokens    = self._get_tokens(sents[idx - 1])
                    prev_token_set = {t.text for t in sents[idx - 1]}
                    name_in_prev   = last_assignee in prev_token_set
                    if name_in_prev:
                        assignee = last_assignee

            if assignee:
                last_assignee = assignee

            next_tokens       = self._get_tokens(sents[idx + 1]) if idx + 1 < len(sents) else None
            deadline          = self._extract_deadline(tokens, next_tokens)
            priority          = self._extract_priority(tokens, deadline)
            has_dep, dep_note = self._check_dependency(tokens)
            description       = self._extract_description(sent, assignee)

            tasks.append(Task(
                id=task_id, description=description,
                assigned_to=assignee or "Unassigned",
                deadline=deadline or "TBD", priority=priority,
                has_dependency=has_dep, dependency_note=dep_note,
                reason=self._member_reason(assignee) if assignee else "",
            ))
            task_id += 1

        return tasks


class TaskAssigner(BaseNLPComponent):

    def __init__(self, team_members: list[TeamMember]):
        self._team       = team_members
        self._team_names = {m.name for m in team_members}

    def _build_workload(self, tasks: list[Task]) -> dict:
        workload = defaultdict(int)
        for task in tasks:
            if task.assigned_to in self._team_names:
                workload[task.assigned_to] += 1
        return workload

    def _is_role_mismatch(self, member: TeamMember, description: str) -> bool:
        desc_tokens = set(description.lower().split())
        role_lower  = member.role.lower()
        for role_keyword, bad_words in ROLE_MISMATCH.items():
            if role_keyword in role_lower:
                if desc_tokens.intersection({w.lower() for w in bad_words}):
                    return True
        return False

    def _best_match(self, task: Task, workload: dict) -> Optional[TeamMember]:
        return max(
            self._team,
            key=lambda m: m.skill_overlap_score(task.description) * 10 - workload[m.name],
            default=None,
        )

    def process(self, tasks: list[Task]) -> list[Task]:
        workload = self._build_workload(tasks)

        for task in tasks:
            current = next((m for m in self._team if m.name == task.assigned_to), None)

            if current:
                if self._is_role_mismatch(current, task.description):
                    best = self._best_match(task, workload)
                    if best and best.name != current.name:
                        workload[current.name] = max(0, workload[current.name] - 1)
                        task.assigned_to = best.name
                        task.reason      = f"{best.role} — better role fit for this task"
                        workload[best.name] += 1
                continue

            best = self._best_match(task, workload)
            if best:
                task.assigned_to = best.name
                task.reason      = task.reason or f"Auto-assigned: {best.role} — best skill match"
                workload[best.name] += 1

        return tasks


class OutputFormatter(BaseNLPComponent):

    def process(self, tasks: list[Task], output_path: str = None) -> str:
        json_str = self._to_json(tasks)
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)
            print(f"[OutputFormatter] JSON saved → {output_path}")
        print(self._to_table(tasks))
        return json_str

    def _to_json(self, tasks: list[Task]) -> str:
        return json.dumps({
            "generated_at": datetime.now().isoformat(),
            "total_tasks":  len(tasks),
            "tasks":        [t.to_dict() for t in tasks],
        }, indent=2)

    def _to_table(self, tasks: list[Task]) -> str:
        if not tasks:
            return "No tasks identified."
        headers = ["#", "Description", "Assigned To", "Deadline", "Priority", "Dependency", "Reason"]
        rows = [
            [
                str(t.id),
                t.description[:45] + ("..." if len(t.description) > 45 else ""),
                t.assigned_to, t.deadline, t.priority,
                (t.dependency_note or "—")[:35],
                (t.reason or "")[:30],
            ]
            for t in tasks
        ]
        widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        sep    = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        fmt    = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
        lines  = ["\n" + "=" * 65, "  IDENTIFIED & ASSIGNED TASKS", "=" * 65,
                  sep, fmt.format(*headers), sep]
        lines += [fmt.format(*row) for row in rows]
        lines += [sep]
        return "\n".join(lines)