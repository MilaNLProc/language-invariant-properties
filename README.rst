
=============================
Language Invariant Properties
=============================


.. image:: https://img.shields.io/pypi/v/language_invariant_properties.svg
        :target: https://pypi.python.org/pypi/language_invariant_properties

.. image:: https://github.com/MilaNLProc/language-invariant-properties/workflows/Python%20package/badge.svg
        :target: https://github.com/MilaNLProc/language-invariant-properties/actions

.. image:: https://readthedocs.org/projects/language-invariant-properties/badge/?version=latest
        :target: https://language-invariant-properties.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Language Invariant Properties (WIP)

* Free software: MIT license
* Documentation: https://language-invariant-properties.readthedocs.io.


Meaning is influenced by a host of factors, among others who says it and when:
"That was a sick performance" changes meaning depending on whether a 16-year-old says it at a concert or a 76-year-old after the opera.
However, here are several properties of text that do (or should) not change when we transform the text. A positive message like "happy birthday!"
should be perceived as positive, regardless of the speaker.  Even when it is translated in Italian (i.e., "buon compleanno!"). The same goes for other properties, it the text has been written by a 25 years old female it should not be perceived as written by an old man after translation. We refer to these properties as
Language Invariant Properties.

Features
--------

In this context, we have three "prediction sets" of interest:

+ original data (O): this is the data we are going to use as a test set. We will use a classifier on it to compute the classifier bias (i.e., how well a classifier does on this test set). Moreover, we will translate this dataset and use another classifier (in another language) on this data.

+ predicted original (PO): this is the data we obtain from running the classifier on O.

+ predicted transformed (PT): this is the data we obtain from running the classifier on the translated O data.

The differences between O-PO and O-PT will tell us if there is a transformation bias or not.

.. code-block:: python

    tp = TrustPilot("english", "italian", "age_cat") # selects source and target language and a property to test

    text_to_translate = tp.get_text_to_translate()["text"].values # get the text to translate

    translated_text = YourTranslator().translate(text_to_translate) # run your translation model

    tp.score(text_to_translate) # get KL and significance

A more concrete example using the Transformer library is given in the following listing:

.. code-block:: python

    from transformers import MarianTokenizer, MarianMTModel
    from transformers import pipeline

    tp = SemEval("spanish", "english", "location/of_the_files/)

    to_translate = tp.get_text_to_translate()["text"].values

    model_name = 'Helsinki-NLP/opus-mt-es-en'

    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation = pipeline("translation_es_to_en", model=model, tokenizer=tokenizer)

    translated = []
    for sent in to_translate:
        translated.append(translation(sent)[0]["translation_text"])


    print(tp.score(translated))

Outputs
-------

The tool is going to output a bunch of metrics that describe the difference between the
original data and the two predicted sets (the predicted on original and the predicted on transformed).

Scores
~~~~~~

Plots
~~~~~

.. image:: https://raw.githubusercontent.com/MilaNLProc/language-invariant-properties/master/img/bias_example.png
   :align: center
   :width: 600px

Tasks
-----

+-------------+-------------------------+-----------------------------+
| DataSet     | Languages               | Tasks                       |
+=============+=========================+=============================+
| TrustPilot  | English, Italian        | Age, Binary Gender          |
+-------------+-------------------------+-----------------------------+
| SemEval19T5 | English, Spanish        | Hate Speech Detection       |
+-------------+-------------------------+-----------------------------+

For SemEval data, interested users should ask access `here <https://github.com/MilaNLProc/language-invariant-properties>`_. Users can place
the files in a folder they like, but they should split the data in a format similar to the one already provided for the
TrustPilot data (train/test folders, a file for each language).

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
