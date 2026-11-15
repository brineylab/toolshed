"""ProteinBERT's model code, vendored from github.com/nadavbra/protein_bert.

Only the three modules the contacts command needs are vendored, from commit
69a1122bd7b590af7506981fb31db06e238a882c (master, 2026-04-07):

- ``tokenization.py``: verbatim.
- ``conv_and_global_attention_model.py``: verbatim except for one change,
  marked in place -- the local (per-token) output head, ``output-seq``, is
  built with ``activation=None`` so the model returns its pre-softmax logits
  instead of probabilities. The weights are untouched.
- this ``__init__.py``: upstream's imports ``existing_model_loading``, the
  interactive downloader, which is not vendored.

Upstream's own loader (``model_generation.load_pretrained_model_from_dump``,
same commit) reads the released dump as
``pickle.load(f) -> (n_annotations, model_weights, optimizer_weights)`` and
sets them on a model built by ``create_model``; the wrapper in ``bin/``
follows exactly that.
"""
