# Story-predictor

This is a novel baseline model
to predict next event sentences (vectors) by RNN and Skip-thoughts encoder.
RNN encodes a sequence of context sentence vecotrs produced by pre-trained skip-thought encoder.

It is developped as a better baseline model in [Story Cloze Test and ROCStories Corpora](http://cs.rochester.edu/nlp/rocstories/).
The great paper propose a novel dataset and tasks, and shows some baseline models and their results.
Among them, one using Skip-thought reaches 0.552, however,
it is too simple to show appropriate performance as a baseline,
e.g., it uses cos similarity between raw skip-thought vectors without any supervised learning, finetuning or transfer learning for ROCStory as DSSM does.
Skip-thought vectors are not designed and learned to use direct cos similarity unlike DSSM (DSSM is reaching 0.585).

As a more proper baseline, I propose a novel (baseline) model using Skip-thought vectors with supervised projection and RNN context encoder,
and it raise this task's baseline to 0.665 on test split and 0.682 on validation split.
I expect it be a more precise evaluation criteria for proposed approaches in future.


## How to run

1. Move to dir `story-predictor-77` and execute `THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python -u skipthoughts/encode_stories.py`  
   (`THEANO_FLAGS='mode=FAST_RUN,device=cpu,floatX=float32' python -u skipthoughts/encode_stories.py` if cpu)  
   This produces vectors (npy) and preprocessed dataset (json) in the current directory.

2. `python -u scripts/train_model.py --load-corpus ./ --save-path data/models --gpu=0 --model-type lstm -b 128 -d 0.2 -u 512 -e 40 --margin 1. -nt 20 > u512.log`


This is implemented with [Chainer](https://github.com/pfnet/chainer).
And it will need:

- Python 2.7 (they may work on other versions)
- [Chainer](https://github.com/pfnet/chainer) 1.7-
- and dependencies for [Chainer](https://github.com/pfnet/chainer)
- in addition to [skip-thoughts](https://github.com/ryankiros/skip-thoughts) (Thanks, @ryankiros)


## Negative example argument

Each of ROCStory's training data is just a 5 sentences story,
not in forms of (4 sentences as context, 1st candidate of the ending, 2nd one, which is the answer)
in the evaluation dataset, Story Cloze Test.
So, as a negative example (candidate of the ending of story),
I use other story's endings or a rewinded sentence, that is, 1-4th sentence included in the story's context.
I expect the 2nd type examples can prevent a classifier from learning only how to discriminate rough domains or characters
by overfitting 1st type examples.

Arg `nt` control the probability of sampling of negative example. See `get_neg` method in `train_model.py` in detail.  
If nt <= 0, sampled from other positive 5th event.  
If 1 <= nt <= 4, sampled from its sequence\[0:nt\] (0,1,...,nt-1).  
If 5 <= nt, sampled from its sequence\[0:nt\] + (nt - 4) of other positive 5th events,
that is, sampled from its sequence\[0:nt\] by probability (4/nt)
and sampled from other positive 5th events by probability ((nt-4)/nt).


## Preprocessed data structure

### Sentence dataset (json dict)

Key is a problem (story) id.
Value is a dict of 'answer' (str) and 'sentences' (list(str)) if test/valid dataset. 'answer' is '1' or '2'.  
If training dataset, a dict of 'title' (str) and 'sentences'.
'sentences' orders are original orders.  
If training, 1,2,3,4,5th sentence.  
If test/valid, 1,2,3,4,1st candidate,2nd candidate.  
(If you want to know which candidate is TRUE/FALSE,
see the 'answer'.)

Examples of keys: ` [u'c33c24e3-c638-4ccb-bea0-cbf4ada0962c', u'02b625bb-bc17-4255-a872-2ccc649dd529', u'e5508db3-e498-4207-80eb-3a6dacb22441', u'26bc6970-8091-4aac-9342-6d0484149753', u'30cf8dc4-0d44-4195-80c5-8de3da99f4c1', u'29fe765b-70a1-4ea1-ba1b-1d8f20dfdc8c', u'0bbf6075-0cb2-4975-81b1-250492df05cf', u'c24717a8-f206-486c-89e7-9978dbb703f5', ...] `


numpy.ndarray.  
If training, its shape is like (45502, 5, 4800). The 1st axis is each story in the sorted story-idx order. The 2nd axis is each sentence in a story. The 3rd axis is each dimension of a sentence vector.  
If test/valid, its shape is like (1871, 6, 4800). The axes are like training. (The 5th and 6th vectors (at the 2nd axis) are candidates vectors.)

4800-dim vectors are based on bi-directional skip-thought vectors. If you want to use only forward-(uni)directional skip-thought vectors, cut and use only the first 2400-dim values.


## Reference

### The method in this repository
This method is not published because I think it is just a simple baseline model and nothing fancy.
However, if you use this baseline model,
I would appreciate if you contact me (http://www.cl.ecei.tohoku.ac.jp/~sosuke.k/)
and just display this repository's url in your paper.
Of cource, you can modify the model better in your paper.
Please ask me anything about this model.


### Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. "Skip-Thought Vectors." NIPS 2015, https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf

Code (including an encoder model, I used it): https://github.com/ryankiros/skip-thoughts

```
@article{kiros2015skip,
  title={Skip-Thought Vectors},
  author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan and Zemel, Richard S and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja},
  journal={arXiv preprint arXiv:1506.06726},
  year={2015}
}
```

### Nasrin Mostafazadeh; Nathanael Chambers; Xiaodong He; Devi Parikh; Dhruv Batra; Lucy Vanderwende; Pushmeet Kohli; James Allen. "A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories." NAACL 2016, http://aclweb.org/anthology/N/N16/N16-1098.pdf

```
@InProceedings{mostafazadeh-EtAl:2016:N16-1,
  author    = {Mostafazadeh, Nasrin  and  Chambers, Nathanael  and  He, Xiaodong  and  Parikh, Devi  and  Batra, Dhruv  and  Vanderwende, Lucy  and  Kohli, Pushmeet  and  Allen, James},
  title     = {A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories},
  booktitle = {Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2016},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics},
  pages     = {839--849},
  url       = {http://www.aclweb.org/anthology/N16-1098}
}
```