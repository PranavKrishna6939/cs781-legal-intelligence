from saul.data.anonymize import anonymize_names_orgs


def test_collapsed_headers_do_not_erase_judgment():
    text = ('PETITIONER: Alice Smith RESPONDENT: Bob Jones DATE OF JUDGMENT: 1 May 2000 '
            'HEADNOTE: The appeal concerns pension rights. JUDGMENT: The appeal is allowed.')
    out = anonymize_names_orgs(text)
    assert out.startswith('PETITIONER: <NAME> RESPONDENT: <NAME> DATE OF JUDGMENT:')
    assert out.endswith('HEADNOTE: The appeal concerns pension rights. JUDGMENT: The appeal is allowed.')


def test_multiline_headers_still_mask_parties():
    text = 'PETITIONER: Alice Smith\nRESPONDENT: Bob Jones\nThe court allowed the appeal.'
    assert anonymize_names_orgs(text) == 'PETITIONER: <NAME>\nRESPONDENT: <NAME>\nThe court allowed the appeal.'


def test_ambiguous_flattened_prose_is_preserved():
    text = 'PETITIONER: Alice Smith filed a petition claiming her pension. The court allowed it.'
    assert anonymize_names_orgs(text) == text
    long = 'PETITIONER: Alice Smith ' + 'Evidence and findings are discussed here. ' * 40
    assert anonymize_names_orgs(long) == long
