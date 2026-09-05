"""Evidence and rules for one-reference title-detail-page explanations."""
import re
from models import JudgeResult

ATTRIBUTES = ('genres', 'tones', 'themes')
GENERATOR_INSTRUCTIONS = (
    'Write one short sentence (at most 35 words) for the recommended title detail page, '
    'reached after selecting it on the homepage. Explain similarity to exactly one supplied '
    'reference title from the member watch history. Quote its exact title with double quotes. '
    'Describe only genre, tone, or theme attributes supported for BOTH titles. '
    'Do not infer liking, viewing frequency, private traits, or additional reference titles. '
    'If there is no valid reference or shared evidence, return an empty text string. ')
JUDGE_INSTRUCTIONS = (
    ' This is a similarity-based explanation for a title detail page. Groundedness requires '
    'that the reference was watched and every asserted shared attribute is supported by BOTH '
    'titles. Relevance requires an explicit comparison to exactly the supplied reference '
    'using shared genre, tone, or theme evidence; a generic recommendation is insufficient. '
    'Reject unsupported statements about either title, invented viewing or liking behavior, '
    'and private member inferences. Clarity requires a single concise message of at most 35 words. '
    'Equivalent natural-language paraphrases of catalog attributes are allowed when supported.')


def normalize(value):
    return ' '.join(value.casefold().split())


def shared_attributes(ex):
    if ex.reference is None:
        return {name: [] for name in ATTRIBUTES}
    shared = {}
    for name in ATTRIBUTES:
        reference = {normalize(v) for v in getattr(ex.reference, name)}
        shared[name] = list(dict.fromkeys(normalize(v) for v in getattr(ex.item, name)
                                         if normalize(v) in reference))
    return shared


def evidence_errors(ex):
    if ex.reference is None:
        return ['A reference title is required.']
    errors = []
    if normalize(ex.reference.title) not in {normalize(t) for t in ex.user.recently_watched}:
        errors.append('Reference title is not in member watch history.')
    if normalize(ex.reference.title) == normalize(ex.item.title):
        errors.append('Reference and recommended title must be different.')
    if not any(shared_attributes(ex).values()):
        errors.append('No shared genre, tone, or theme evidence.')
    return errors


def structure_errors(ex):
    errors = evidence_errors(ex)
    quoted = re.findall(r'[“"]([^”"]+)[”"]', ex.explanation)
    if [q.removesuffix('.') for q in quoted] != [ex.reference.title.removesuffix('.')]:
        errors.append('Quote exactly one reference title, matching the supplied title.')
    if not ex.explanation.strip() or len(ex.explanation.split()) > 35:
        errors.append('Explanation must be nonempty and at most 35 words.')
    return errors


def enforce_structure(ex, result):
    errors = structure_errors(ex)
    if errors:
        if evidence_errors(ex):
            result.groundedness = 0
        result.relevance = 0
        if not ex.explanation.strip() or len(ex.explanation.split()) > 35:
            result.clarity = 0
        result.passed = False
        result.unsupported_claims.extend(errors)
        result.rationale += ' ' + ' '.join(errors)
    return result


def render_explanation(ex):
    if evidence_errors(ex):
        # Empty output is rejected by the guardrail, which serves its neutral fallback.
        return ''
    shared = shared_attributes(ex)
    description = ', '.join(shared['tones'][:2])
    genre = ' and '.join(shared['genres'][:1])
    description = ' '.join(v for v in (description, genre) if v)
    themes = ' and '.join(shared['themes'][:2])
    if description:
        article = 'An' if description[0].lower() in 'aeiou' else 'A'
        description = f'{article} {description}'
        if not genre:
            description += ' story'
    else:
        description = 'A story'
    if themes:
        description += ' about ' + themes
    return f'{description}, much like “{ex.reference.title}.”'


def evaluate_demo(ex, decouple_relevance=False):
    """Conservative vocabulary checker for fixtures; real mode judges paraphrases."""
    errors = structure_errors(ex)
    shared = shared_attributes(ex)
    text = normalize(ex.explanation.replace('“', '"').replace('”', '"'))
    # Remove the one authorized reference before checking attribute vocabulary.
    body = re.sub(r'"' + re.escape(normalize(ex.reference.title)) + r'\.?"', '', text)
    allowed = set('a an and about much like story with the both share shares'.split())
    for values in shared.values():
        for value in values:
            allowed.update(re.findall(r'\w+', value))
    unexpected = sorted(set(re.findall(r'\w+', body)) - allowed)
    grounding_errors = evidence_errors(ex)
    if unexpected:
        grounding_errors.append('Unsupported attribute or claim')
        errors.append('Unsupported claims or wording: ' + ', '.join(unexpected))
    mentions_attribute = any(normalize(value) in body for values in shared.values() for value in values)
    relevance = 2 if mentions_attribute and normalize(ex.reference.title) in text else 0
    quoted = re.findall(r'[“"]([^”"]+)[”"]', ex.explanation)
    if [q.removesuffix('.') for q in quoted] != [ex.reference.title.removesuffix('.')]:
        relevance = 0
    if grounding_errors and relevance and not decouple_relevance:
        relevance = 1
    clarity = 2 if ex.explanation.strip() and len(ex.explanation.split()) <= 35 else 0
    return JudgeResult(0 if grounding_errors else 2, relevance, 2, clarity,
                       not errors and relevance >= 1 and clarity >= 1,
                       unsupported_claims=errors, rationale=' '.join(errors) or 'Both titles support the shared attributes.')
