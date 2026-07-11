def normalize_event_filter(events):
    if events is None:
        return None
    if isinstance(events, str):
        events = (events,)
    events = frozenset(events)
    if not all(isinstance(event, str) for event in events):
        raise TypeError("event filters must contain strings")
    return events


def should_write_event(event, events) -> bool:
    return events is None or event in events
