=============================
Language Invariant Properties
=============================


.. image:: https://img.shields.io/pypi/v/language_invariant_properties.svg
        :target: https://pypi.python.org/pypi/language_invariant_properties

.. image:: https://img.shields.io/travis/vinid/language_invariant_properties.svg
        :target: https://travis-ci.com/vinid/language_invariant_properties

.. image:: https://readthedocs.org/projects/language-invariant-properties/badge/?version=latest
        :target: https://language-invariant-properties.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Language Invariant Properties (WIP)


* Free software: MIT license
* Documentation: https://language-invariant-properties.readthedocs.io.


Features
--------

.. code-block:: python

    tp = TrustPilot("english", "italian", "age_cat") # selects source and target language and a property to test

    text_to_translate = tp.get_text_to_translate()["text"].values # get the text to translate

    translated_text = YourTranslator().translate(text_to_translate) # run your translation model

    tp.score(text_to_translate) # get KL and significance

A more concrete example using the Transformer library is given in the following listing:

.. code-block:: python

    from transformers import MarianTokenizer, MarianMTModel
    from transformers import pipeline

    tp = SemEval("spanish", "english")

    to_translate = (tp.get_text_to_translate()["text"].values)

    model_name = 'Helsinki-NLP/opus-mt-es-en'

    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation = pipeline("translation_es_to_en", model=model, tokenizer=tokenizer)

    translated = []
    for a in to_translate:
        translated.append(translation(a)[0]["translation_text"])


    print(tp.score(translated))


Tasks
-----

+-------------+-------------------------+--------------------+
| DataSet     | Languages               | Tasks              |
+=============+=========================+====================+
| TrustPilot  | English, Italian        | Age, Binary Gender |
+-------------+-------------------------+--------------------+
| SemEval19T5 | English, Spanish        | Hate Speech        |
+-------------+-------------------------+--------------------+

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
