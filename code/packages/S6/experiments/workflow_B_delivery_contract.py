"""Static visibility semantics: no labels, hidden records, or counts in prompts."""
VERSION='whole-source-filtered-view-1'
SELECTOR='''The coordinator has completed all frozen source reads for this workflow.
The supplied records contain every result of those reads. A nonempty list of
objects is represented by its individual objects, with original tool, arguments
and result_index. Other results, including an empty list [], are retained whole.
This is complete coverage of the frozen reads, not a claim that those reads answer
every possible question about the environment. Select records using their content;
record IDs and indices are identifiers, not a global count or an answer key.'''
RECEIVER='''You receive only the source records assigned to your task. This may be a
filtered subset of the coordinator's original read results. Original tool, arguments
and result_index identify the origin of a delivered record; they do not certify
that you received every result of that read. A missing record does not prove that
the underlying information or event is absent. An explicit empty list [] is an
observed empty result, not a missing source. Do not invent omitted records or infer
complete lists solely from record IDs. If the supplied evidence is insufficient,
state that limitation. No additional source-reading tools are available.'''

def contract(workflow,audience):
    version=workflow.get('delivery_contract_version')
    if version is None:return None
    if version!=VERSION:raise ValueError('Unknown delivery contract version')
    if audience=='selector':return SELECTOR
    if audience=='receiver':return RECEIVER
    raise ValueError('Unknown audience')
