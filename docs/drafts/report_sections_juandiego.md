# Report draft – Transformer, comparative analysis, Turing machines, conclusions

Owner: Juan Diego Balanta. Sections to merge into the shared report (T9).
Values written as `{{...}}` are placeholders filled from `outputs/metrics/comparison_table.md`
once every model has its final run. Target length for these sections: about 1.25 pages
including Table 2 and one figure.

---

## 4. Transformer extension (DistilBERT)

As an extension we fine-tuned DistilBERT (`distilbert-base-uncased`) [Sanh et al., 2019], a
6-layer Transformer encoder distilled from BERT-base [Devlin et al., 2019] that keeps about 97 %
of BERT's language-understanding performance with 40 % fewer parameters (≈66 M). Unlike the
Dense, RNN and LSTM models, which learn their representations from the 800 training comments only,
DistilBERT starts from weights pre-trained on English Wikipedia and BookCorpus and only adapts
them to our task (transfer learning).

**Input.** The model reads the *raw* comment. Its WordPiece tokenizer splits rare words and
slang into sub-word units (e.g. *"lol"*, *"upsets"* → *"upset"*, *"##s"*), and we keep
stopwords and punctuation because the pre-trained model relies on them; removing them would drop
negations such as *"not"* and *"don't"*, which flip the sentiment of a sentence. Sequences are
truncated/padded to {{MAX_LEN}} tokens, which covers 99 % of the comments.

**Architecture and training.** A classification head (linear 768→768, ReLU, dropout 0.2, linear
768→2) is placed on the final hidden state of the `[CLS]` token, and the whole network is
fine-tuned with cross-entropy, AdamW (weight decay 0.01), a linear learning-rate schedule with 10 %
warm-up and gradient clipping at 1.0. Hyperparameters were selected with a manual grid on a
stratified validation split (15 % of train): learning rate ∈ {2e-5, 3e-5, 5e-5} × batch size ∈
{16, 32}, up to 4 epochs, keeping the epoch with the best validation macro-F1. The selected
configuration was learning rate {{LR}}, batch size {{BS}} and {{EPOCHS}} epochs; it trained in
{{TIME}} s on a Colab T4 GPU. The test set was used once, for the final evaluation.

## 5. Comparative analysis

**Table 2.** Test performance of all models (best value per column in bold). Train F1 and the
gap train − test F1 are reported to diagnose overfitting.

{{TABLE_2  ← paste outputs/metrics/comparison_table.md, or \input{comparison_table.tex}}}

**Performance.** Both Dummy baselines stay at chance level (accuracy ≈ 0.50, κ ≈ 0), which
confirms that the balanced 500/500 dataset gives no shortcut through the class prior. All neural
models clearly beat the baseline. {{One–two sentences: ranking by test F1 and κ; improvement of the
best recurrent model over Dense; improvement of DistilBERT over the best recurrent model, in
points of F1 and κ.}} Cohen's κ is the most informative metric here: with balanced classes
accuracy and κ carry the same ranking, but κ makes explicit how far each model is from chance
(κ = 0) and from perfect agreement (κ = 1).

**Overfitting.** {{Train vs test F1 per model, see Figure X.}} With only 680 training comments
the models trained from scratch can memorise the training set — their train F1 approaches 1 while
the test F1 stays {{GAP}} points lower — whereas DistilBERT updates already-useful weights for a
few epochs and generalises better from the same data.

**Training complexity.** The Dense network trains in seconds because it sees each comment as an
unordered bag of {{TF-IDF / averaged embedding}} features and has no recurrence. The RNN and LSTM
must unroll over every time step, so they cannot parallelise over the sequence; the LSTM has four
times the recurrent parameters of the vanilla RNN (one set per gate). DistilBERT is two orders
of magnitude larger (≈66 M parameters) and needs a GPU, but its self-attention processes all
tokens of a comment in parallel, so with 800 short comments it fine-tunes in minutes.

**Interpretability.** The Dense model is the easiest to inspect: with TF-IDF inputs, the
weights connected to each word give a direct, if approximate, notion of which words push towards
each class. The hidden state of an RNN/LSTM is a dense vector with no direct reading, and its
gates can only be analysed indirectly. DistilBERT provides attention weights; we plot the
attention of `[CLS]` in the last layer (Figure Y), which typically concentrates on sentiment words
and emojis, but attention weights are not faithful explanations by themselves [Jain & Wallace,
2019], so we use them only as qualitative evidence.

**Errors.** {{Two or three misclassified test comments from the error-analysis cell; typical
cases are irony/sarcasm, negated praise and comments that are short or depend on Reddit context.}}

## 6. Relation to Turing machines: memory, sequence processing and computability

A Turing machine (TM) separates a *finite control* from an *unbounded tape* that it can read,
write and revisit in any order [Turing, 1936]. The models in this project differ precisely in
what plays the role of the tape.

**Memory.** The Dense network has no memory at all: it is a fixed function of a bag of words,
closer to a combinational circuit than to an automaton, and it cannot distinguish *"good, not bad"*
from *"bad, not good"*. A vanilla RNN reads the comment once, left to right, and summarises
everything it has seen in a fixed-size hidden state $h_t = \tanh(W h_{t-1} + U x_t + b)$. This is
the behaviour of a finite-state transducer with a very large, continuous state: it has a read-only
input tape that only moves right and no writable work tape. In practice, its memory is short,
because gradients that flow back through many applications of $W$ vanish or explode
[Bengio et al., 1994]. The LSTM [Hochreiter & Schmidhuber, 1997] adds a cell state $c_t$ with
*write* (input gate), *erase* (forget gate) and *read* (output gate) operations, a learned and
differentiable version of a TM's read/write head acting on a single memory cell; additive updates
of $c_t$ keep gradients alive over longer distances. Neural Turing Machines [Graves et al., 2014]
take this analogy literally by attaching an external memory matrix with attention-based read and
write heads. A Transformer has no recurrent state: at every layer each token attends to all others,
so the whole comment acts as a random-access tape that can be read in any order in a single step,
but the number of steps (layers) is fixed in advance.

**Sequence processing.** RNNs and LSTMs process the sequence like a TM head moving one cell per
step, so the distance between two related words (e.g. *"not"* and *"funny"*) becomes a number of
steps the information must survive. Self-attention connects every pair of positions directly
(path length 1) [Vaswani et al., 2017], which explains why it handles negation and long-range cues
better and why it parallelises on a GPU, at the cost of $O(n^2)$ attention per layer.

**Computability.** Theoretical results place these architectures surprisingly high in the
Chomsky hierarchy: RNNs with rational weights and *unbounded precision* can simulate any TM
[Siegelmann & Sontag, 1995], and Transformers with arbitrary precision and an unbounded number of
decoding steps are also Turing-complete [Pérez et al., 2021]. These results depend on idealised
assumptions. With finite precision and a fixed input length, every model we trained computes a
function over a finite set of inputs; it behaves as a (huge) finite automaton and never needs
more than its fixed memory [Weiss et al., 2018]. Our task, mapping a comment of at most 30 words to
one of two labels, is trivially decidable, and the differences between the models are not about
*what* can be computed but about inductive bias, how easily the right function is *learned* from
800 examples and how well it generalises. Finally, as with Rice's theorem for TMs, there is no
general procedure to decide non-trivial semantic properties of what an arbitrary trained network
computes, so models are evaluated empirically, as in this project, on held-out data.

## 7. Conclusions and future work

{{Fill after results.}} Expected structure (4–6 sentences):

1. Best model and by how much it beats the baseline and the other architectures (F1, κ).
2. Recurrence helped/did not help over the bag-of-words Dense model on comments of ~13 words,
   and the LSTM vs vanilla RNN difference, explained by memory/gradient flow.
3. Pre-training (DistilBERT) is the largest single improvement with only 800 labelled comments,
   at a much higher computational cost.
4. Main error source: irony and context-dependent comments.

**Future work.** (i) Use the full positive/negative subset (≈ 25 k unique comments) instead of
1 000 to measure how the gap between recurrent models and DistilBERT changes with data;
(ii) initialise the RNN/LSTM with pre-trained word embeddings (GloVe/fastText) and try
bidirectional layers; (iii) move to the original multi-label 28-class problem; (iv) apply
k-fold cross-validation to report variance, since 200 test comments give confidence intervals of
about ±6 accuracy points; (v) use faithful attribution methods (e.g. integrated gradients)
instead of raw attention for interpretability.

## References (to add to the shared list)

- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient
  descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157–166.
- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020).
  GoEmotions: A dataset of fine-grained emotions. *Proc. ACL 2020*, 4040–4054.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep
  bidirectional Transformers for language understanding. *Proc. NAACL-HLT 2019*, 4171–4186.
- Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing machines. arXiv:1410.5401.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8),
  1735–1780.
- Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *Proc. NAACL-HLT 2019*,
  3543–3556.
- Pérez, J., Barceló, P., & Marinkovic, J. (2021). Attention is Turing-complete. *Journal of
  Machine Learning Research*, 22(75), 1–35.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT:
  smaller, faster, cheaper and lighter. arXiv:1910.01108.
- Siegelmann, H. T., & Sontag, E. D. (1995). On the computational power of neural nets.
  *Journal of Computer and System Sciences*, 50(1), 132–150.
- Turing, A. M. (1936). On computable numbers, with an application to the Entscheidungsproblem.
  *Proc. London Mathematical Society*, s2-42(1), 230–265.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., &
  Polosukhin, I. (2017). Attention is all you need. *Advances in NeurIPS 30*, 5998–6008.
- Weiss, G., Goldberg, Y., & Yahav, E. (2018). On the practical computational power of finite
  precision RNNs for language recognition. *Proc. ACL 2018*, 740–745.
