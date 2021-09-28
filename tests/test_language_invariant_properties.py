#!/usr/bin/env python

"""Tests for `language_invariant_properties` package."""

import os
from language_invariant_properties.lip import TrustPilot


def test_trust_pilot_loading():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dir = f"{root_dir}/../language_invariant_properties/data/trustpilot"
    tp = TrustPilot("italian", "english", "age_cat", dir)
    _ = tp.get_text_to_translate()["text"].values




