"""Optional request-scoped reuse of unchanged global hybrid retrieval output."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class RetrievalRequest:
    question: str
    enabled: bool
    retrieve_calls: int = 0
    initial_output: object = None
    events: list = field(default_factory=list)


class InitialGlobalCache:
    def __init__(self):
        self._request = ContextVar('initial_global_retrieval_request', default=None)

    @contextmanager
    def request(self, question, *, enabled=False):
        state = RetrievalRequest(question, enabled)
        token = self._request.set(state)
        try:
            yield state
        finally:
            self._request.reset(token)

    def begin_retrieve(self, question, routed_sources, queries):
        state = self._request.get()
        if state is None:
            return False
        state.retrieve_calls += 1
        return bool(state.enabled and state.retrieve_calls > 1 and routed_sources
                    and question == state.question and queries == [state.question])

    def call(self, query, delegate, *, allow_reuse=False):
        state = self._request.get()
        if state is None or not state.enabled:
            return delegate(query)
        start = perf_counter()
        hit = bool(allow_reuse and query == state.question and state.initial_output is not None)
        if hit:
            output = deepcopy(state.initial_output)
        else:
            output = delegate(query)
            # Capture only the initial retrieval's original query. A structured
            # first retrieval has no global output; later calls cannot invent it.
            if state.retrieve_calls == 1 and query == state.question:
                state.initial_output = deepcopy(output)
        state.events.append({'query': query, 'cache_hit': hit,
            'global_models_executed': not hit, 'seconds': round(perf_counter() - start, 6)})
        return output
